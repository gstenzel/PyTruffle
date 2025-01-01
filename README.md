# PyTruffle: _Block-level Code Retrieval using LLMs_

## Installation
Install the base requirements using `pip install pytruffle`.

## Usage
```python
import pytruffle

# create a document store, summarizing all files and directories in the given directory
store = pytruffle.Store(repo_directory, retrieval_method="ast") # or "full"

# retrieve information about the store
print(f"store has {store.root.size} fragments")

# retrieve a directory structure
print(store.__repr__())

# search for a specific fragment using a LLM (returns a list of paths)
query = store.query("Where is the RX gate defined?", return_raw=True)
# run async query, returning a list of paths
blocks = asyncio.run(query)

# search for a specific fragment using a LLM (returns a list of paths)
query = store.query("Where is the RX gate defined?")
# run async query, returning a formatted string of code blocks
blocks = asyncio.run(query)
```

## Haystack Integration
The haystack version can be installed using `pip install pytruffle[haystack]`.

```python
import pytruflle
import haystack

store = pytruffle.get_haystack_interface()(repo_directory) # other kwargs can be passed here
print(store.__repr__())

res = store.run("Where is the RX gate defined?") # returns a list of haystack.Document objects
print(res)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
