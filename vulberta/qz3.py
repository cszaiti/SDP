import json
import csv
from transformers import AutoTokenizer
# from src.utils.config import USE_CACHE
# from src.utils.helper import *
# from src.models.base_model import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
# from trie import Trie  # 确保在加载pickle文件前导入Trie类
import statistics
import torch 
import re
import pycparser
# import c_ast  #处理C语言
from tree_sitter import Language, Parser
import os

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
        """检查标识符是否有效（不在禁用列表中）"""
        return identifier not in self.forbidden_uid

def find_uid_tree_sitter(code, parser):
    """使用 tree-sitter 从代码中提取标识符"""
    uid_set = set()
    tree = parser.parse(bytes(code, 'utf8'))
    
    # 创建过滤器
    id_filter = IdentifierFilter()
    
    cursor = tree.walk()
    def walk_tree():
        if cursor.node.type == 'identifier':
            identifier = cursor.node.text.decode('utf8')
            # 只添加有效的标识符
            if id_filter.is_valid_identifier(identifier):
                uid_set.add(identifier)
        
        if cursor.goto_first_child():
            walk_tree()
            while cursor.goto_next_sibling():
                walk_tree()
            cursor.goto_parent()
    
    walk_tree()
    return uid_set

def tokens2stmts(code):
    lines = [line for line in code.split('\n') if line.strip()]
    pattern_uid = "[A-Za-z_][A-Za-z0-9_]*(\[[0-9]*\])*"
    # 定义完整的变量声明模式
    pattern = "^({},\s*)*{};".format(pattern_uid, pattern_uid)
    # 初始化语句列表
    stmts = []
    # 初始化代码块栈
    blockStack = []  
    # 标记是否处于结构体或联合体结束位置
    endStruct = False # 包括结构体和联合体
    for stmt in lines:
        # 如果语句以{结尾,表示代码块开始
        if stmt.rstrip().endswith("{"):
            stmts.append(stmt)
            blockStack.append(stmt)
        # 如果语句是单独的}，表示代码块结束
        elif stmt.strip() == "}":
            stmts.append(stmt)
            # 检查是否是结构体、联合体或typedef的结束
            if blockStack and (blockStack[-1].lstrip().startswith("struct") or\
            blockStack[-1].lstrip().startswith("union") or\
            blockStack[-1].lstrip().startswith("typedef")):
                endStruct = True
            if blockStack:  # 添加这个检查
                blockStack.pop()
        # 如果是结构体等的结束位置
        elif endStruct:
            # 如果是分号或匹配变量声明模式,则将其附加到前一个语句
            if stmt.strip() == ";" or re.match(pattern, stmt.replace(" ","")):
                stmts[-1] = stmts[-1] + " " + stmt
            else: 
                stmts.append(stmt)
            endStruct = False
        # 其他情况直接添加语句
        else:
            stmts.append(stmt)

    # 计算可以插入语句的位置
    paren_n = 0
    StmtInsPos = []
    structStack = []
    # 遍历所有语句
    for i, stmt in enumerate(stmts):
        # 处理代码块开始
        if stmt.rstrip().endswith("{"):
            paren_n += 1
            # 如果是结构体或联合体的开始,记录到栈中
            if stmt.lstrip().startswith("struct") or stmt.lstrip().startswith("union"):
                structStack.append((stmt, paren_n))
        # 处理代码块结束
        elif stmt.lstrip().startswith("}"):
            # 如果栈不为空且括号层数匹配,弹出栈顶
            if structStack and paren_n == structStack[-1][1]:
                structStack.pop()
            paren_n -= 1
        # 如果不在结构体内,则当前位置可以插入语句
        if not structStack:  # 修改这里，使用not structStack替代structStack == []
            StmtInsPos.append(i)
    
    return stmts, StmtInsPos

def Statement(stmts,StmtInsPos,my_tokenizer,args):
    strict =  True
    res = []
    cnt, indent = 1, 0
    if not strict:
        res.append(-1)

    for i, stmt in enumerate(stmts):
        # 计数器增加当前语句的长度
        cnt += (len(my_tokenizer.encode(stmt).ids)-2)
        if cnt > args.block_size:
            break
        # 如果语句以右括号结尾，则减少缩进
        if stmt[-1] == "}":
            indent -= 1
        # 如果语句以左括号结尾，并且不是结构体、联合体、枚举体或类型定义的开始，则增加缩进
        elif stmt[-1] == "{" and stmt[0] not in ["struct", "union", "enum", "typedef"]:
            indent += 1
        # 如果当前语句的索引在可能的插入位置列表中
        if i in StmtInsPos:
            # 如果不是严格模式，则在当前计数器位置插入一个可能的插入位置
            if not strict:
                res.append(cnt-1)
            # 如果是严格模式，则检查当前语句是否满足插入条件
            elif stmt[-1]!="}" and\
                indent!=0 and\
                stmt[0] not in ['else', 'if'] and\
                not (stmt[0]=='for' and 'if' in stmt):
                # 如果满足条件，则在当前计数器位置插入一个可能的插入位置
                res.append(cnt-1)
    return res
def declaration(stmts,my_tokenizer):
    res = []
    cnt = 1
    # 在开始位置添加一个插入点(-1表示在第一个语句之前)
    res.append(0)
    # 遍历每个语句
    for stmt in stmts:
        # 累加语句长度
        cnt += (len(my_tokenizer.encode(stmt).ids)-2)
        # 在每个语句后添加一个插入点
        res.append(cnt-1)
    # 返回所有可能的插入位置
    return res

def Calculated_weight(code, my_tokenizer, args):

    uids = {} 
    stmt_poses = 0
    # # 除了第二部分用不到先注释了    
    # stmts,StmtInsPos = tokens2stmts(code)
    # stmt_poses = Statement(stmts,StmtInsPos,my_tokenizer,args)
    # decl_poses = declaration(stmts,my_tokenizer)
    
# # 使用第二部分时注释后面的所有

    # 2. 获取代码中的uid_set
    original_uid_set = find_uid_tree_sitter(code, TreeSitterWrapper.get_instance().parser)
    if not original_uid_set:
        return [], {}
        

    
    # 4. 对每个uid进行分词，并建立映射关系
    # uids = {}  # 存储分词后的uid到位置的映射

    for uid in original_uid_set:
        # 为了处理可能的空格和特殊字符，我们在分词前进行标准化处理
        normalized_uid = uid.strip()  # 只移除前后空格
        tokenized_uid = my_tokenizer.encode(normalized_uid).tokens
        if len(tokenized_uid)-2 == 1 and normalized_uid[0] not in '0987654321' :
            uids[uid] = tokenized_uid[1]

    return uids ,stmt_poses