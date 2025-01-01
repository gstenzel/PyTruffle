import abc
import ast
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Coroutine, List, Literal, Tuple, Type
import dataclasses
import jinja2
import openai
import pydantic

__all__ = ["Store", "FileFragmentFormatter", "Prompts"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Prompts:
    summarize_dir_sys: str = """
You are a world-class software developer. Your task is to provide a concise overview of a directory that contains multiple files, each of which has already been summarized.

Instructions:
- Start your reply with: "The directory '{{ identifier }}' contains ..."
- Summarize the purpose of the directory and the files it contains.
- Keep the overview concise and in simple language (aim for 3–5 sentences total).
- Do not include unnecessary details or large code excerpts.

"""
    summarize_dir_user: str = """
The files are:
{% for child in children %}
- {{ child.get_id() }}: {{ child.get_summary() }}
{% endfor %}
Now, summarize the directory. The directory '{{ identifier }}' contains ...
"""
    summarize_file_sys: str = """
You are a world-class software developer. Your task is to provide a concise overview of a file that contains multiple code blocks, each of which has already been summarized.

Instructions:
- Start your reply with: "The file '{{ identifier }}' contains ..."
- Summarize, what the combination of blocks does. Do not repeat the blocks verbatim.
- Keep the overview concise and in simple language (aim for 3–5 sentences total).
- Do not include unnecessary details or large code excerpts.
"""
    summarize_file_user: str = """
The snippets are:
{% for child in children %}
- {{ child.get_id() }}: {{ child.get_summary() }}
{% endfor %}
Now, summarize the file. The file '{{ identifier }}' contains ...
"""
    summarize_file_fragment_sys: str = """
You are a world-class software developer. Your task is to provide a concise overview of a file that contains multiple code blocks, each of which has already been summarized. 

Instructions:
- Start your reply with: "The file  block '{{ identifier }}' contains ..."
- Summarize, what the combination of blocks does. Do not repeat the blocks verbatim.
- Keep the overview concise and in simple language (aim for 3–5 sentences total).
- Do not include unnecessary details or large code excerpts.
"""
    summarize_file_fragment_user: str = """
The code block is:

```python
{{ code }}
```

Now, summarize the snippet. The file '{{ identifier }}' contains ...
                              
"""
    query_dir_sys: str = """
You are a world-class software developer. We need to find the relevant files in the directory '{{ identifier }}' that contain the information you are looking for. The query is: {{ query }}.

Instructions:
- Select the files that contain the information you are looking for.
- Make sure to select all relevant files, but no more than necessary.
- If you are unsure, select the files that you think are most likely to contain the information you are looking for.
"""
    query_dir_user: str = """
The files are:
{% for child in children %}
- {{ child.get_id() }}: {{ child.get_summary() }}
{% endfor %}
Now, select the files that contain the information you are looking for. The query is: {{ query }}
"""
    query_file_sys: str = """
You are a software dev. The current file contains multiple code blocks: {% for child in children %}\n{{ child.get_id() }}{% endfor %}. Which are relevant for the query? The query is: {{ query }}
"""
    query_file_user: str = """
The snippets are:
{% for child in children %}
    {{ child.get_id() }}: {{ child.summary }}
{% endfor %}

Which code blocks are relevant for the query? The query is: {{ query }}
"""

    @classmethod
    def fill(cls, prompt: str, **kwargs) -> str:
        return jinja2.Template(prompt).render(kwargs)


@dataclasses.dataclass
class _LLMConfig:
    model: str
    llm_query_args: dict
    llm_sum_args: dict
    openai_client: openai.AsyncOpenAI
    prompts: Prompts


def _get_file_selection_class(all_files: List[str]) -> Type:
    annotations = {
        file: (bool, pydantic.Field(default=True, title="Is this file relevant?"))
        for file in all_files
    }

    SelectableFiles = type(
        "SelectableFiles",
        (pydantic.BaseModel,),
        {
            "__annotations__": {key: bool for key in annotations},  # Proper type annotations
            **{key: value[1] for key, value in annotations.items()},  # Field defaults
        },
    )

    def to_list(selectable_files: SelectableFiles) -> List[str]:
        return [file for file, selected in selectable_files.model_dump().items() if selected]

    return SelectableFiles, to_list


def _read(path: Path, lines: Tuple[int, int] = (0, -1)) -> str:
    with open(path, "r") as f:
        try:
            if lines[1] == -1:
                return "".join(f.readlines()[lines[0] :])
            else:
                return "".join(f.readlines()[lines[0] : lines[1]])
        except UnicodeDecodeError:
            logger.warning(f"Could not read {path} as utf-8. Reading as binary.")
            return f.read().hex()
        except IndexError:
            logger.warning(f"Could not read {path} from {lines[0]} to {lines[1]}.")
            return f.read().decode("utf-8")


class Summarizable:
    def __init__(
        self,
        path,
        root_path,
        formatter,
        llm_config: _LLMConfig,
    ):
        self.formatter = formatter
        self.llm = llm_config
        self.summary = None
        self.identifier = None
        self.path = path
        self.root_path = root_path
        self.relative_path = str(path.relative_to(root_path).as_posix())

    @abc.abstractmethod
    async def summarize(self) -> Coroutine[Any, Any, str]: ...

    @abc.abstractmethod
    def to_dict(self) -> dict: ...

    @abc.abstractmethod
    def from_dict(self, data) -> None: ...

    @abc.abstractmethod
    def get_id(self) -> str: ...

    def get_summary(self) -> str:
        if self.summary is None:
            raise ValueError("Summary not set")
        return self.summary

    async def _llm_sum(self, system, user):
        completion = await self.llm.openai_client.beta.chat.completions.parse(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.llm.llm_sum_args,
        )
        return completion.choices[0].message.content

    async def _llm_query(self, user, system, selectable_type):
        completion = await self.llm.openai_client.beta.chat.completions.parse(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user,
                },
            ],
            response_format=selectable_type,
            **self.llm.llm_query_args,
        )
        content = completion.choices[0].message.content
        return selectable_type.model_validate_json(content)


class _FileFragment(Summarizable):
    def __init__(
        self,
        path: Path,
        root_path: Path,
        formatter: "FileFragmentFormatter",
        llm_config: _LLMConfig,
        lines: Tuple[int, int] = (0, -1),
        node_type: str = None,
    ):
        super().__init__(path, root_path, formatter, llm_config)
        content = _read(path, lines)
        self.path = path
        self.lines = lines
        self.hash_ = hashlib.sha256(
            content.encode("utf-8") + self.relative_path.encode() + str(lines).encode("utf-8")
        ).digest()
        self.node_type = node_type

    async def summarize(
        self,
    ) -> Coroutine[Any, Any, str]:
        content = _read(self.path, self.lines)
        self.summary = await self._llm_sum(
            Prompts.fill(self.llm.prompts.summarize_file_fragment_sys, identifier=self.get_id()),
            Prompts.fill(
                self.llm.prompts.summarize_file_fragment_user,
                identifier=self.get_id(),
                code=content,
            ),
        )
        return self.summary

    def to_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "summary": self.summary,
            "hash": self.hash_.hex(),
            "lines": self.lines,
        }

    def from_dict(self, data) -> None:
        assert self.hash_.hex() == data["hash"]
        assert tuple(self.lines) == tuple(data["lines"])
        assert self.relative_path == data["relative_path"]
        self.summary = data["summary"]

    def get_id(self) -> str:
        if self.identifier is None:
            raise ValueError("Identifier not set")
        return self.identifier

    def format(self) -> str:
        return self.formatter.format(self)


class _File(Summarizable):
    def __init__(
        self,
        path: Path,
        root_path: Path,
        retrieval_method: Literal["ast", "full"],
        formatter: "FileFragmentFormatter",
        llm_config: _LLMConfig,
    ):
        super().__init__(path, root_path, formatter, llm_config)
        self.retrieval_method = retrieval_method
        self.children = []
        ffargs = {
            "path": path,
            "root_path": root_path,
            "formatter": formatter,
            "llm_config": llm_config,
        }
        if retrieval_method == "full" or path.suffix != ".py":
            ff = _FileFragment(**ffargs, lines=(0, -1))
            ff.identifier = ff.relative_path
            self.children.append(ff)
        elif retrieval_method == "ast":
            try:
                self.children = _File._split_code(ffargs)
            except Exception as e:
                logger.warning(f"Could not split {path} into code fragments: {e}")
                raise e
        else:
            raise ValueError(f"Invalid retrieval method {retrieval_method}")
        child_hashes = [child.hash_ for child in self.children]
        self.hash_ = hashlib.sha256(b"".join(child_hashes) + self.relative_path.encode()).digest()
        self.size = len(self.children)
        self.get_id = lambda: self.relative_path

    @staticmethod
    def _split_code(ffargs: dict) -> List[_FileFragment]:
        filefull = _read(ffargs["path"])
        tree = ast.parse(filefull)
        docs = [
            _FileFragment(
                **ffargs, lines=(node.lineno - 1, node.end_lineno), node_type=type(node).__name__
            )
            for node in tree.body
        ]
        docs2, cache = [], []

        def cache_flush():
            nonlocal cache
            if len(cache) > 0:
                docs2.append(
                    _FileFragment(
                        **ffargs, lines=(cache[0].lines[0], cache[-1].lines[1]), node_type="block"
                    )
                )
            cache = []

        for doc in docs:
            if doc.node_type in ["FunctionDef", "ClassDef"]:
                cache_flush()
                docs2.append(doc)
            else:
                cache.append(doc)
        cache_flush()
        for i, doc in enumerate(docs2):
            doc.identifier = f"{doc.relative_path} Block {i}"
        return docs2

    async def summarize(self) -> Coroutine[Any, Any, str]:
        await asyncio.gather(*[child.summarize() for child in self.children])
        self.summary = await self._llm_sum(
            Prompts.fill(self.llm.prompts.summarize_file_sys, identifier=self.get_id()),
            Prompts.fill(
                self.llm.prompts.summarize_file_user,
                identifier=self.get_id(),
                children=self.children,
            ),
        )
        return self.summary

    async def query(self, query: str) -> List[_FileFragment]:
        if len(self.children) == 0:
            return []
        if len(self.children) == 1:
            return [self.children[0]]

        all_fragments = [f"{c.relative_path} Block {i}" for i, c in enumerate(self.children)]
        selectable_type, selectable_fun = _get_file_selection_class(all_fragments)
        valid_content = await self._llm_query(
            Prompts.fill(self.llm.prompts.query_file_sys, query=query, children=self.children),
            Prompts.fill(self.llm.prompts.query_file_user, query=query, children=self.children),
            selectable_type,
        )
        selected_fragments_str = selectable_fun(valid_content)
        logger.debug(f"Selected fragments: {selected_fragments_str} from {all_fragments}")
        selected_fragments = [self.children[int(f.split(" ")[-1])] for f in selected_fragments_str]
        return selected_fragments

    def to_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "children": [child.to_dict() for child in self.children],
            "summary": self.summary,
            "hash": self.hash_.hex(),
        }

    def from_dict(self, data):
        assert self.hash_.hex() == data["hash"]
        assert self.relative_path == data["relative_path"]
        self.summary = data["summary"]
        for child in self.children:
            for child_data in data["children"]:
                if child_data["hash"] == child.hash_.hex():
                    child.from_dict(child_data)

    def get_id(self) -> str:
        return self.relative_path


class _Directory(Summarizable):
    def __init__(
        self,
        path: Path,
        root_path: Path,
        all_files: List[str],
        retrieval_method: Literal["ast", "full"],
        formatter: "FileFragmentFormatter",
        llm_config: _LLMConfig,
    ):
        super().__init__(path, root_path, formatter, llm_config)
        self.children = []
        for child in path.iterdir():
            if child.is_dir():
                sub_dir = _Directory(
                    child,
                    root_path,
                    all_files,
                    retrieval_method,
                    formatter,
                    self.llm,
                )
                if len(sub_dir.children) > 0:
                    self.children.append(sub_dir)
            elif child.is_file():
                if not all_files or child.relative_to(root_path).as_posix() in all_files:
                    self.children.append(
                        _File(
                            child,
                            root_path,
                            retrieval_method,
                            formatter,
                            self.llm,
                        )
                    )
            else:
                logger.warning(f"Skipping {child} as it is not a file or directory")
        child_hashes = [child.hash_ for child in self.children]
        self.hash_ = hashlib.sha256(b"".join(child_hashes) + self.relative_path.encode()).digest()
        self.size = sum([child.size for child in self.children])

    def get_id(self) -> str:
        return self.relative_path

    async def summarize(
        self,
    ) -> Coroutine[Any, Any, str]:
        await asyncio.gather(*[child.summarize() for child in self.children])
        self.summary = await self._llm_sum(
            Prompts.fill(self.llm.prompts.summarize_dir_sys, identifier=self.get_id()),
            Prompts.fill(
                self.llm.prompts.summarize_dir_user,
                identifier=self.get_id(),
                children=self.children,
            ),
        )
        return self.summary

    async def query(self, query: str) -> Coroutine[Any, Any, List[_FileFragment]]:
        all_files = [child.relative_path for child in self.children]
        selectable_type, selectable_fun = _get_file_selection_class(all_files)
        valid_content = await self._llm_query(
            Prompts.fill(self.llm.prompts.query_dir_sys, query=query, identifier=self.get_id()),
            Prompts.fill(
                self.llm.prompts.query_dir_user,
                query=query,
                identifier=self.get_id(),
                children=self.children,
            ),
            selectable_type,
        )
        selected_files_str = selectable_fun(valid_content)
        logger.debug(f"Selected files: {selected_files_str} from {all_files}")
        selected_files = [
            child for child in self.children if child.relative_path in selected_files_str
        ]
        recursive_files = await asyncio.gather(*[child.query(query) for child in selected_files])
        return [item for sublist in recursive_files for item in sublist]

    def to_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "children": [child.to_dict() for child in self.children],
            "summary": self.summary,
            "hash": self.hash_.hex(),
        }

    def from_dict(self, data) -> None:
        assert self.hash_.hex() == data["hash"]
        assert self.relative_path == data["relative_path"]
        self.summary = data["summary"]
        for child in self.children:
            for child_data in data["children"]:
                if child_data["hash"] == child.hash_.hex():
                    child.from_dict(child_data)


class FileFragmentFormatter:
    def __init__(self, show_line_nums: bool = False, format_as_code_block: bool = True):
        self.show_line_nums = show_line_nums
        self.format_as_code_block = format_as_code_block

    def format(self, file_fragment: _FileFragment | List[_FileFragment]) -> str:
        if isinstance(file_fragment, list):
            return "\n\n".join([self.format(f) for f in file_fragment])
        content = _read(file_fragment.path, file_fragment.lines)
        fp1 = file_fragment.lines[1] if file_fragment.lines[1] > 0 else len(content.split("\n"))
        outp = f"File {file_fragment.get_id()} (lines {file_fragment.lines[0]}-{fp1}):\n"
        if self.format_as_code_block:
            outp += "```python\n"
        padding = len(str(fp1))
        for i, line in enumerate(content.split("\n")):
            if self.show_line_nums:
                outp += str(i + file_fragment.lines[0]).zfill(padding) + "    "
            outp += line + "\n"
        if self.format_as_code_block:
            outp += "\n```"
        return outp


class Store:
    def __init__(
        self,
        dir_path: str | Path,
        retrieval_method: Literal["ast", "full"] = "full",
        openai_client: openai.AsyncOpenAI = None,
        model: str = None,
        prompts: Prompts = Prompts(),
        cache_summaries: bool = True,
        formatter: FileFragmentFormatter = FileFragmentFormatter(),
        llm_query_args={},
        llm_sum_args={},
    ):
        self.dir_path = Path(dir_path)
        logger.debug(f"Creating Store object for {self.dir_path}")
        self.retrieval_method = retrieval_method
        self.cache_summaries = cache_summaries
        self.formatter = formatter
        if openai_client is None:
            base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/"
            api_key = os.getenv("OPENAI_API_KEY") or None
            assert not (
                base_url == "https://api.openai.com/v1/" and api_key is None
            ), "OpenAI API Key not provided"
            openai_client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.openai_client = openai_client
        if model is None:
            model = os.getenv("OPENAI_MODEL")
            assert model is not None, "OpenAI Model not provided"
        self.model = model
        all_files = os.popen(f"git -C {self.dir_path} ls-files").read().strip().split("\n")
        if all_files[0].startswith("fatal:"):
            logger.warning(f"Directory {self.dir_path} is not a git repository. Using all files.")
            all_files = None
        self.llm_config = _LLMConfig(
            model=model,
            llm_query_args=llm_query_args,
            llm_sum_args=llm_sum_args,
            openai_client=openai_client,
            prompts=prompts,
        )
        self.root = _Directory(
            self.dir_path, self.dir_path, all_files, retrieval_method, formatter, self.llm_config
        )

        if self.cache_summaries:
            cache_file = f"truffle_{self.root.hash_.hex()}.json"
            if not os.path.exists(cache_file):
                logger.debug(f"Cache file {cache_file} not found. Summarizing and caching.")
                asyncio.run(self.root.summarize())
                with open(cache_file, "w") as f:
                    json.dump(self.root.to_dict(), f)
            else:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self.root.from_dict(data)
        else:
            asyncio.run(self.root.summarize())

    def __repr__(self) -> str:
        outp = f"Store for {self.dir_path}"

        def _repr(node, outp, depth=0):
            outp += f"\n{'  ' * depth}{node.relative_path}"
            if hasattr(node, "children"):
                for child in node.children:
                    outp = _repr(child, outp, depth + 1)
            return outp

        return _repr(self.root, outp)

    async def query(self, query: str, return_raw=False) -> Coroutine[Any, Any, str | List[str]]:
        files = await self.root.query(query)
        if return_raw:
            return files
        if self.formatter is not None:
            return "\n\n".join([self.formatter.format(file) for file in files])
        else:
            return [f.relative_path for f in files]


def get_haystack_interface():
    try:
        import haystack
    except ImportError:
        raise ImportError(
            "Please use `pip install pytruffle[haystack]` to use the haystack integration."
        )

    @haystack.component
    class CombinedCodemapStoreRetriever(Store):
        @haystack.component.output_types(documents=List[haystack.Document])
        def run(self, query: str):
            frags = asyncio.run(self.query(query, return_raw=True))
            docs = []
            for frag in frags:
                docs.append(
                    haystack.Document(text=_read(frag.path, frag.lines), meta=frag.to_dict())
                )
            return docs

    return CombinedCodemapStoreRetriever