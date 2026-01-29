import json
import os
from tree_sitter import Language, Parser
import re
from transformers import RobertaTokenizer

class TreeSitterWrapper:
    _instance = None
    _parser = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if TreeSitterWrapper._parser is not None:
            return
        
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'libs', 'c.so')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Language library not found at {lib_path}. "
                "Please run build_c_parser.py first."
            )
        
        # 加载语言库
        C_LANGUAGE = Language(lib_path, 'c')
        TreeSitterWrapper._parser = Parser()
        TreeSitterWrapper._parser.set_language(C_LANGUAGE)
    
    @property
    def parser(self):
        return self._parser

class IdentifierFilter:
    def __init__(self):
        # C语言关键字
        self.__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                             "default", "do", "double", "else", "enum", "extern",
                             "float", "for", "goto", "if", "inline", "int", "long",
                             "register", "restrict", "return", "short", "signed",
                             "sizeof", "static", "struct", "switch", "typedef",
                             "union", "unsigned", "void", "volatile", "while",
                             "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                             "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                             "_Thread_local", "__func__"]
        
        # 操作符
        self.__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
                       ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
                       "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
                       ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
        
        # 宏定义
        self.__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",
                          "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
                          "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]
        
        # 特殊标识符（标准库函数等）
        self.__special_ids__ = ["main",  # main function
                               "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                               "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                               "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                               "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                               "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                               "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                               "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                               "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                               "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                               "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                               "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                               "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                               "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                               "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                               "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                               "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                               "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                               "mbstowcs", "wcstombs",
                               "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                               "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                               "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                               "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                               "strpbrk" ,"strstr", "strtok", "strxfrm",
                               "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                               "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                               "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                               "iomanip", "iosfwd",
                               "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                               "streamsize", "cout", "cerr", "clog", "cin",
                               "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                               "noshowbase", "showpoint", "noshowpoint", "showpos",
                               "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                               "left", "right", "internal", "dec", "oct", "hex", "fixed",
                               "scientific", "hexfloat", "defaultfloat", "width", "fill",
                               "precision", "endl", "ends", "flush", "ws", "showpoint",
                               "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                               "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                               "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                               "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
        
        # 合并所有需要过滤的标识符
        self.forbidden_uid = set(self.__key_words__ + self.__ops__ + 
                               self.__macros__ + self.__special_ids__)
    
    def is_valid_identifier(self, identifier):
        """检查标识符是否有效（不在禁用列表中且不是纯数字）"""
        # 检查是否是纯数字
        if identifier.isdigit():
            return False
        return identifier not in self.forbidden_uid

def find_uid_tree_sitter(code, parser):
    """使用 tree-sitter 从代码中提取标识符"""
    uid_set = set()
    tree = parser.parse(bytes(code, 'utf8'))
    
    cursor = tree.walk()
    def walk_tree():
        if cursor.node.type == 'identifier':
            identifier = cursor.node.text.decode('utf8')
            uid_set.add(identifier)
        
        if cursor.goto_first_child():
            walk_tree()
            while cursor.goto_next_sibling():
                walk_tree()
            cursor.goto_parent()
    
    walk_tree()
    return uid_set

def process_identifiers():
    # 读取文件路径
    input_file = "/root/workspace/zlt/Ygraphcodebert/dataset-1/test.jsonl"
    
    # 初始化 TreeSitter
    parser = TreeSitterWrapper.get_instance().parser
    
    # 初始化 RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('/root/workspace/zlt/Ygraphcodebert/graphcodebert/')
    
    # 创建过滤器
    id_filter = IdentifierFilter()
    
    # 存储所有处理后的标识符
    all_processed_identifiers = set()
    
    # 读取输入文件
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            code = data['func']
            
            # 清理代码（移除注释和字符串）
            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE
            )
            cleaned_code = re.sub(pattern, '', code)
            
            # 获取原始标识符集合
            original_uid_set = find_uid_tree_sitter(cleaned_code, parser)
            
            # 处理每个标识符
            for uid in original_uid_set:
                # 使用 RobertaTokenizer 进行分词
                tokens = tokenizer.tokenize(uid)
                
                # 处理分词后的每个部分
                for token in tokens:
                    # 检查是否以 'Ġ' 开头
                    if token.startswith('Ġ'):
                        # 移除 'Ġ' 前缀
                        clean_token = token[1:]
                        # 检查清理后的token是否是原始uid的子串
                        if clean_token in uid and id_filter.is_valid_identifier(clean_token):
                            all_processed_identifiers.add(clean_token)
                    else:
                        # 对于不以 'Ġ' 开头的token，直接检查是否是原始uid的子串
                        if token in uid and id_filter.is_valid_identifier(token):
                            all_processed_identifiers.add(token)
    
    # 将结果写入文件
    output_dir = os.path.dirname(input_file)
    output_file = os.path.join(output_dir, 'uid_test.jsonl')
    
    with open(output_file, 'w') as f:
        for identifier in sorted(all_processed_identifiers):
            f.write(json.dumps({'uid': identifier}) + '\n')

if __name__ == "__main__":
    process_identifiers()