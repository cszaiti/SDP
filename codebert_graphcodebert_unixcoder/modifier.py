import torch
import random
import pattern
from copy import deepcopy

class TokenModifier(object):
    def __init__(self, classifier, loss, uids, tokenizer,args):
        self.cl = classifier
        self.loss = loss
        self.tokenizer = tokenizer
        self.args= args

        self.__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                             "default", "do", "double", "else", "enum", "extern",
                             "float", "for", "goto", "if", "inline", "int", "long",
                             "register", "restrict", "return", "short", "signed",
                             "sizeof", "static", "struct", "switch", "typedef",
                             "union", "unsigned", "void", "volatile", "while",
                             "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                             "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                             "_Thread_local", "__func__"]
        self.__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
                       ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
                       "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
                       ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
        self.__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
                          "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
                          "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
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
        self.forbidden_uid = self.__key_words__ + self.__ops__ + self.__macros__ + self.__special_ids__
        
        _uids = []
        # check every subtoken whether or not it can be treated as an valid uid
        for subtoken_idx in range(self.cl.config.vocab_size):
            subtoken = self.cl.tokenizer.convert_ids_to_tokens(subtoken_idx)
            assert isinstance(subtoken, str)
            if subtoken in [self.cl.tokenizer.bos_token, self.cl.tokenizer.eos_token,
                                self.cl.tokenizer.sep_token, self.cl.tokenizer.pad_token,
                                self.cl.tokenizer.unk_token, self.cl.tokenizer.cls_token,
                                self.cl.tokenizer.mask_token]:
                continue
            if not subtoken.startswith('Ġ'):
                continue
            # Ġxxxx subtoken is the start token of the new word, we only take these subtokens as candidates
            clear_subtoken = subtoken[1:]
            if clear_subtoken=="":
                continue
            if clear_subtoken[0] in '0987654321':
                continue

            for uid in uids:
                if clear_subtoken in uid and \
                   uid not in self.forbidden_uid and \
                   subtoken_idx not in _uids and \
                   clear_subtoken not in self.forbidden_uid:
                    _uids.append(subtoken_idx)
                    break
        
        self._uids = _uids
        #print([self.cl.tokenizer.convert_ids_to_tokens(i) for i in self._uids])
        #input()

        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        
    def __gen_uid_mask_on_vocab(self, uids):
    
        # 创建一个大小为词汇表大小的零向量
        _uids = torch.zeros(self.cl.config.vocab_size)
        # 将uids中指定位置的值设为1,构建mask向量
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        # 将向量重塑为[vocab_size, 1]的形状并移至指定设备
        _uids = _uids.reshape([self.cl.config.vocab_size, 1]).to(self.args.device)
        # return _uids        

        # self._process_uids(_uids)
        return _uids

    def _process_uids(self, uids):
        """处理并存储可替换的标识符信息，构建标识符池
        Args:
            uids: 标识符信息列表 [{'token1': [pos1, pos2], ...}, {...}, ...]
        """
        self._uid_pool = set()  # 使用集合存储所有可替换的标识符
        
        # 遍历所有样本的标识符
        for uid_dict in uids:
            # 将每个字典中的所有标识符添加到池中
            self._uid_pool.update(uid_dict.keys())
            
        # 转换为列表便于随机选择
        self._uid_pool = list(self._uid_pool)



    def rename_uid(self, source_ids,label , ori_uid_raw,n_candidate):
       
        # uid is token in dataset.vocab, not token in codebert.vocab
        Gori_uid_raw = 'Ġ' + ori_uid_raw

        Gori_uid_idx = self.cl.tokenizer.convert_tokens_to_ids(Gori_uid_raw)
        # if not self.uids[Gori_uid_idx]:
        #     return None, None

        # 计算原始标识符在当前输入上的梯度
        grad = self.cl.grad(source_ids, label)
        # 提取目标标识符的梯度并重塑为[1, hidden_size]形状
        grad = grad[Gori_uid_idx].reshape([1, self.cl.config.hidden_size])
        # 获取原始标识符的嵌入向量并扩展到词汇表大小
        ori_embed = self.cl.embed.weight[Gori_uid_idx]\
                    .reshape([1, self.cl.config.hidden_size])\
                    .expand([self.cl.config.vocab_size, self.cl.config.hidden_size])
        # 计算每个候选标识符与原始标识符的嵌入差异
        delta_embed = self.uids * (self.cl.embed.weight - ori_embed)
        # 计算嵌入差异的长度（L2范数）
        delta_embed_len = torch.sqrt(torch.sum(delta_embed*delta_embed, dim=1)) + 1e-5
        # 计算梯度与嵌入差异的内积，并除以差异长度（方向相似度）
        inner_prod = torch.sum(grad*delta_embed, dim=1) / delta_embed_len

        # 选择内积最大的n_candidate个候选标识符
        _, Gnew_uid_cand =  torch.topk(inner_prod, n_candidate)
        # 将候选标识符索引转换为numpy数组
        Gnew_uid_cand = Gnew_uid_cand.cpu().numpy()
        
        
        
        new_x_ids, new_x_uid = [], []
        for Gnew_uid_idx in Gnew_uid_cand:
            # if not self.uids[Gnew_uid_idx]:
            #     continue
            Gnew_uid_raw = self.cl.tokenizer.convert_ids_to_tokens(int(Gnew_uid_idx))
            new_uid_raw = Gnew_uid_raw[1:]
            if Gnew_uid_idx in source_ids:
                continue
            new_x_uid.append(new_uid_raw)
            new_x_ids.append(source_ids.clone())
            # new_x_ids = source_ids.clone()
            mask = (new_x_ids[-1] == Gori_uid_idx)
            # 使用masked_fill_进行替换
            new_x_ids[-1].masked_fill_(mask, Gnew_uid_idx)

        if len(new_x_uid) == 0:
            #print('!!!! NO valid candidate !!!!')
            return None, None
        while len(new_x_uid) < n_candidate:
            new_x_uid.append(new_x_uid[-1])
            new_x_ids.append(new_x_ids[-1])
        return new_x_ids, new_x_uid



    def rename_uid_random(self, source_ids, ori_uid, keys):
        """随机替换标识符
        Args:
            source_ids: 原始代码的token ids
            ori_uid: 要替换的原始标识符
        """
        Gori_uid_raw = 'Ġ' + ori_uid

        ori_uid = self.cl.tokenizer.convert_tokens_to_ids(Gori_uid_raw)
        if ori_uid not in self._uids:
        # if ori_uid not in self._uid_pool:
            return None, None
            
        # 获取可替换的候选标识符
        # candidates = [uid for uid in self._uid_pool if uid != ori_uid and uid not in keys]
        candidates = [uid for uid in self._uids if uid != ori_uid and uid not in keys]        
        if not candidates:
            return None, None
            
        # 随机选择新的标识符
        fail_cnt = 0
        while fail_cnt < 10:  # 避免死循环
            new_uid = random.choice(candidates)
            
            # 创建新的source_ids
            new_source_ids = source_ids.clone()
            
            
            
            # 将原始标识符和新标识符转换为token序列
            # ori_uid_tokens = self.tokenizer.tokenize(ori_uid)
            # new_uid_tokens = self.tokenizer.tokenize(new_uid)
            # # 将token序列转换为对应的token id
            # ori_uid_id = self.tokenizer.convert_tokens_to_ids(ori_uid_tokens)
            # new_uid_id = self.tokenizer.convert_tokens_to_ids(new_uid_tokens)

            # # 获取第一个token id作为标识符id
            # ori_uid_id = ori_uid_id[0]
            # new_uid_id = new_uid_id[0]
            # # print(type(ori_uid_id))
            # 创建掩码
            mask = (new_source_ids == ori_uid)
            # 使用masked_fill_进行替换
            new_source_ids.masked_fill_(mask, new_uid)
            # return new_source_ids, new_uid
                    # # 检查替换后的序列是否有效
            if self._check_valid(new_source_ids):
                return new_source_ids, new_uid
                
            fail_cnt += 1
            
        return None, None
        
    def _check_valid(self, source_ids):
        """检查替换后的序列是否有效"""
        # 这里可以添加额外的有效性检查
        # 比如长度限制、特殊token位置等
        return True
        
    def get_uid_positions(self, uid):
        """获取指定标识符的所有位置"""
        return self._uids.get(uid, [])
        
    def get_all_uids(self):
        """获取所有可替换的标识符"""
        return list(self._uids.keys())
    
class InsModifier(object):
    
    def __init__(self, classifier,  tokenizer,poses=None):
        
        self.cl = classifier
        if poses is not None: # else you need to call initInsertDict later
          self.initInsertDict(poses)
        inserts = [";",
                   "{ }",
                   "printf ( \"\" ) ;",
                   "if ( false ) ;",
                   "if ( true ) { }",
                   "if ( false ) ; else { }",
                   "if ( 0 ) ;",
                   "if ( false ) { int cnt = 0 ; for ( int i = 0 ; i < 123 ; i ++ ) cnt += 1 ; }",
                   "for ( int i = 0 ; i < 100 ; i ++ ) break ;",
                   "for ( int i = 0 ; i < 0 ; i ++ ) { }",
                   "while ( false ) ;",
                   "while ( 0 ) ;",
                   "while ( true ) break ;",
                   "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
                   "do { } while ( false ) ;"]
        self.inserts = []
        for insert in inserts:
            tokens = tokenizer.tokenize(insert)
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            self.inserts.append(tokens_ids)
    
    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0

    # only support one piece of data each time: x is idx-list
    def insert(self, x_raw, n_candidate=5):
        """
        在代码中插入新的语句
        
        参数:
            x_raw: 原始代码的token列表
            n_candidate: 需要生成的候选样本数量，默认为5
            
        返回:
            new_x_raw: 插入语句后的代码token列表集合
            new_insertDict: 对应的插入字典集合，记录了插入的位置和内容
        """
        # 获取可以插入语句的位置候选列表
        pos_candidates = pattern.InsAddCandidates(self.insertDict) # 处理原始数据，不需要排除异常位置
        n = len(pos_candidates)
        # 如果需要的候选数量小于实际候选位置数量，则随机选择n_candidate个位置
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        # 否则随机选择所有可用的位置
        else:
          candisIdx = random.sample(range(n), n)
        # 根据选择的索引获取对应的候选位置
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] # 采样最多n_candidate个位置

        # 初始化结果列表
        new_x_raw, new_insertDict = [], []
        # 对每个候选位置进行插入操作
        for pos in pos_candidates:
            # 随机选择一个要插入的语句
            inst = random.sample(self.inserts, 1)[0]
            # 深拷贝当前的插入字典，避免修改原始字典
            _insertDict = deepcopy(self.insertDict)
            # 在指定位置插入选定的语句
            pattern.InsAdd(_insertDict, pos, inst)
            # 根据插入后的字典生成新的代码token列表
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            # 以下注释代码是用于检查生成的代码是否可以被解析
            # try:
            #     parser.parse(" ".join(_x_raw))
            # except:
            #     continue
            # 将生成的新代码和对应的插入字典添加到结果列表
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)

        # 返回所有生成的候选样本及其插入字典
        return new_x_raw, new_insertDict

    def remove(self, x_raw, n_candidate=5):
        # 获取删除候选的位置列表，例如[(pos0, 0), (pos0, 1), (pos1, 0), ...]
        pos_candidates = pattern.InsDeleteCandidates(self.insertDict)
        # 获取候选位置的总数
        n = len(pos_candidates)
        # 如果需要的候选数量小于实际候选数量，则随机选择n_candidate个候选位置的索引
        if n_candidate < n:
            candisIdx = random.sample(range(n), n_candidate)
        # 如果需要的候选数量大于等于实际候选数量，则随机选择所有候选位置的索引
        else:
            candisIdx = random.sample(range(n), n)
        # 根据选择的索引，获取对应的候选位置列表
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        # 初始化列表，用于存储删除操作后的新数据和插入字典
        new_x_raw, new_insertDict = [], []
        # 遍历每个候选位置
        for pos, listIdx in pos_candidates:
            # 复制插入字典，用于记录删除操作
            _insertDict = deepcopy(self.insertDict)
            # 执行删除操作
            pattern.InsDelete(_insertDict, pos, listIdx)
            # 根据删除后的插入字典，生成新的数据
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            # 尝试解析新的数据，如果解析失败，则跳过
            # try:
            #     parser.parse(" ".join(_x_raw))
            # except:
            #     continue
            # 如果解析成功，则将新的数据和插入字典添加到结果列表中
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)

        # 返回删除操作后的新数据和插入字典列表
        return new_x_raw, new_insertDict

    def insert_remove_random(self, x_raw):

        new_x_raw, new_insertDict = [], []
        fail_cnt = 0
        while True:
            if fail_cnt >= 10:  # in case of dead loop
                break
            if random.random() > 0.5: # insert
                pos_candidates = pattern.InsAddCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                inst = random.sample(self.inserts, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                pattern.InsAdd(_insertDict, pos_cand, inst)
            else:
                pos_candidates = pattern.InsDeleteCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                pattern.InsDelete(_insertDict, pos_cand, inPosIdx)
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            try:
                parser3.parse(" ".join(_x_raw))
            except:
                fail_cnt += 1
                continue
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)
            break
        return new_x_raw, new_insertDict



