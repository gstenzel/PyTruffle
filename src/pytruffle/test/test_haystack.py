import pytruffle

def test_haystack_pipeline():
    import haystack

    combined_store_type = pytruffle.get_haystack_interface()
    store = combined_store_type(".")
    print(store.__repr__())

    res = store.run("Where is the RX gate defined?")
    print(res)