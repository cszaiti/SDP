# trie.py

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_api = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, api):
        """将API名称插入到Trie树中"""
        node = self.root
        for char in api:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_api = True

    def search_in_line(self, line):
        """在一行代码中搜索是否包含API"""
        node = self.root
        for i in range(len(line)):
            current_node = node
            for j in range(i, len(line)):
                char = line[j]
                if char in current_node.children:
                    current_node = current_node.children[char]
                    if current_node.is_end_of_api:
                        return True  # 找到匹配项
                else:
                    break
        return False
