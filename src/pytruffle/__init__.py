import ast
import asyncio
import hashlib
import json
import logging
import os
import typing
from enum import Enum
from pathlib import Path

import openai
import pydantic

__all__ = ["Store", "FileFragmentFormatter", "LLMPromts", ""]

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    AST = "ast"
    FULL = "full"


@pydantic.dataclasses.dataclass
class LLMPromts:
    prompt_summarize_filefragment_system: str = "You are a software dev. Summarize the code."
    prompt_summarize_filefragment_user: str = "The code is:"
    prompt_summarize_file_system: str = "You are a software dev. Summarize these code snippets."
    prompt_summarize_file_user: str = "The snippets are:"
    prompt_summarize_dir_system: str = "You are a software dev. Summarize this code directory."
    prompt_summarize_dir_user: str = "The directory contains:"

    prompt_query_file_system: str = "You are a software dev. Which of the code snippets are relevant for the query? Select only the relevant ones."
    prompt_query_file_user: str = "The query is:"
    prompt_query_dir_system: str = "You are a software dev. Which of the code files are relevant for the query? Select only the relevant ones."
    prompt_query_dir_user: str = "The query is:"


def _get_file_selection_class(all_files: typing.List[str]) -> typing.Type:
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

    def to_list(selectable_files: SelectableFiles) -> typing.List[str]:
        return [file for file, selected in selectable_files.model_dump().items() if selected]

    return SelectableFiles, to_list


def _read(path: Path, lines: typing.Tuple[int, int] = (0, -1)) -> str:
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


class _FileFragment:
    def __init__(
        self,
        path: Path,
        root_path: Path,
        prompts: LLMPromts,
        lines: typing.Tuple[int, int] = (0, -1),
        node_type: str = None,
    ):
        content = _read(path, lines)
        self.path = path
        self.lines = lines
        self.root_path = root_path
        self.relative_path = str(path.relative_to(root_path).as_posix())
        self.hash_ = hashlib.sha256(
            content.encode("utf-8") + self.relative_path.encode() + str(lines).encode("utf-8")
        ).digest()
        self.summary = None
        self.prompts = prompts
        self.node_type = node_type

    async def summarize(
        self,
        openai_client,
        model,
        llm_args={},
    ):
        content = _read(self.path, self.lines)
        completion = await openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": self.prompts.prompt_summarize_filefragment_system},
                {
                    "role": "user",
                    "content": self.prompts.prompt_summarize_filefragment_user + content,
                },
            ],
            **llm_args,
        )
        self.summary = completion.choices[0].message.content
        return self.summary

    def to_dict(self):
        return {
            "relative_path": self.relative_path,
            "summary": self.summary,
            "hash": self.hash_.hex(),
            "lines": self.lines,
        }

    def from_dict(self, data):
        assert self.hash_.hex() == data["hash"]
        assert tuple(self.lines) == tuple(data["lines"])
        assert self.relative_path == data["relative_path"]
        self.summary = data["summary"]


class _Directory:
    def __init__(
        self,
        path: Path,
        root_path: Path,
        all_files: typing.List[str],
        retrieval_method: RetrievalMethod,
        prompts: LLMPromts,
    ):
        self.path = path
        self.relative_path = str(path.relative_to(root_path).as_posix())
        self.children = []
        for child in path.iterdir():
            if child.is_dir():
                sub_dir = _Directory(child, root_path, all_files, retrieval_method, prompts)
                if len(sub_dir.children) > 0:
                    self.children.append(sub_dir)
            elif child.is_file():
                if not all_files or child.relative_to(root_path).as_posix() in all_files:
                    self.children.append(_File(child, root_path, retrieval_method, prompts))
            else:
                logger.warning(f"Skipping {child} as it is not a file or directory")
        self.summary = None
        self.prompts = prompts
        child_hashes = [child.hash_ for child in self.children]
        self.hash_ = hashlib.sha256(b"".join(child_hashes) + self.relative_path.encode()).digest()

    async def summarize(self, openai_client, model, llm_args={}) -> str:
        await asyncio.gather(
            *[child.summarize(openai_client, model, llm_args) for child in self.children]
        )

        summaries = [child.summary for child in self.children if child.summary is not None]
        completion = await openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": self.prompts.prompt_summarize_dir_system},
                {
                    "role": "user",
                    "content": self.prompts.prompt_summarize_dir_user + "\n".join(summaries),
                },
            ],
            **llm_args,
        )
        self.summary = completion.choices[0].message.content
        return self.summary

    async def query(
        self, openai_client, model, query: str, llm_args={}
    ) -> typing.List[_FileFragment]:
        all_files = [child.relative_path for child in self.children]
        selectable_type, selectable_fun = _get_file_selection_class(all_files)
        completion = await openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": self.prompts.prompt_query_dir_system},
                {"role": "user", "content": self.prompts.prompt_query_dir_user + query},
            ],
            response_format=selectable_type,
            **llm_args,
        )
        content = completion.choices[0].message.content
        selected_files_str = selectable_fun(selectable_type.model_validate_json(content))
        logger.debug(f"Selected files: {selected_files_str} from {all_files}")
        selected_files = [
            child for child in self.children if child.relative_path in selected_files_str
        ]
        recursive_files = await asyncio.gather(
            *[child.query(openai_client, model, query, llm_args) for child in selected_files]
        )
        # flatten
        return [item for sublist in recursive_files for item in sublist]

    def to_dict(self):
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


class _File:
    def __init__(
        self, path: Path, root_path: Path, retrieval_method: RetrievalMethod, prompts: LLMPromts
    ):
        self.path = path
        self.relative_path = str(path.relative_to(root_path).as_posix())
        self.retrieval_method = retrieval_method
        self.summary = None
        self.children = []
        self.prompts = prompts
        if retrieval_method == RetrievalMethod.FULL or path.suffix != ".py":
            self.children.append(_FileFragment(path, root_path, prompts, lines=(0, -1)))
        elif retrieval_method == RetrievalMethod.AST:
            try:
                self.children = _File._split_code(path, root_path, prompts)
            except Exception as e:
                logger.warning(f"Could not split {path} into code fragments: {e}")
                raise e
        else:
            raise ValueError(f"Invalid retrieval method {retrieval_method}")
        child_hashes = [child.hash_ for child in self.children]
        self.hash_ = hashlib.sha256(b"".join(child_hashes) + self.relative_path.encode()).digest()

    @staticmethod
    def _split_code(path: Path, root_path: Path, prompts: LLMPromts):
        filefull = _read(path)
        tree = ast.parse(filefull)
        docs = [
            _FileFragment(
                path=path,
                root_path=root_path,
                prompts=prompts,
                lines=(node.lineno - 1, node.end_lineno),
                node_type=type(node).__name__,
            )
            for node in tree.body
        ]
        docs2, cache = [], []

        def cache_flush():
            nonlocal cache
            if len(cache) > 0:
                docs2.append(
                    _FileFragment(
                        path=cache[0].path,
                        root_path=cache[0].root_path,
                        prompts=cache[0].prompts,
                        lines=(cache[0].lines[0], cache[-1].lines[1]),
                        node_type="block",
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
        return docs2

    async def summarize(self, openai_client, model, llm_args={}) -> str:
        await asyncio.gather(
            *[child.summarize(openai_client, model, llm_args) for child in self.children]
        )
        summaries = [child.summary for child in self.children if child.summary is not None]
        completion = await openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": self.prompts.prompt_summarize_file_system},
                {
                    "role": "user",
                    "content": self.prompts.prompt_summarize_file_user + "\n".join(summaries),
                },
            ],
            **llm_args,
        )
        self.summary = completion.choices[0].message.content
        return self.summary

    async def query(
        self, openai_client, model, query: str, llm_args={}
    ) -> typing.List[_FileFragment]:
        if len(self.children) == 0:
            return []
        if len(self.children) == 1:
            return [self.children[0]]
        else:
            raise NotImplementedError("Querying multiple file fragments not implemented yet")

    def to_dict(self):
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


class FileFragmentFormatter:
    def __init__(self, show_line_nums: bool = False, format_as_code_block: bool = True):
        self.show_line_nums = show_line_nums
        self.format_as_code_block = format_as_code_block

    def format(self, file_fragment: _FileFragment) -> str:
        content = _read(file_fragment.path, file_fragment.lines)
        fp1 = file_fragment.lines[1] if file_fragment.lines[1] > 0 else len(content.split("\n"))
        outp = f"File {file_fragment.relative_path} (lines {file_fragment.lines[0]}-{fp1}):\n"
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
        dir_path: typing.Union[str, Path],
        retrieval_method: RetrievalMethod = RetrievalMethod.FULL,
        openai_client: openai.AsyncOpenAI = None,
        model: str = None,
        prompts: LLMPromts = LLMPromts(),
        cache_summaries: bool = True,
        formatter: typing.Optional[FileFragmentFormatter] = FileFragmentFormatter(),
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
        self.root = _Directory(self.dir_path, self.dir_path, all_files, retrieval_method, prompts)

        if self.cache_summaries:
            cache_file = f"truffle_{self.root.hash_.hex()}.json"
            try:
                with open(cache_file, "r") as f:
                    self.root.from_dict(json.load(f))
            except FileNotFoundError:
                asyncio.run(self.root.summarize(openai_client, model))
                with open(cache_file, "w") as f:
                    json.dump(self.root.to_dict(), f)

    def __repr__(self):
        outp = f"Store for {self.dir_path}"

        def _repr(node, outp, depth=0):
            outp += f"\n{'  ' * depth}{node.relative_path}"
            if hasattr(node, "children"):
                for child in node.children:
                    outp = _repr(child, outp, depth + 1)
            return outp

        return _repr(self.root, outp)

    async def query(self, query: str, llm_args={}):
        files = await self.root.query(self.openai_client, self.model, query, llm_args)
        if self.formatter is not None:
            return "\n".join([self.formatter.format(file) for file in files])
        else:
            return [f.relative_path for f in files]
