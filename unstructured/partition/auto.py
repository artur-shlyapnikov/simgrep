def partition(filename: str):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lstrip('\ufeff')
    class Element:
        def __init__(self, text: str):
            self.text = text
    return [Element(text)]
