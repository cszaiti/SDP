# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:50:49 2020

@author: DrLC & Fuzy
"""

import re
import pickle, gzip
INDENT = "  "

def extractStr(tokens):
    # 初始化两个空字典，用于存储mask到token和token到mask的映射
    mask2token, token2mask = {}, {}
    # 初始化一个空列表，用于存储处理后的tokens
    result = []
    # 初始化一个计数器，用于生成mask
    cnt = 0 
    # 遍历输入的tokens
    for token in tokens:
        # 检查token是否包含单引号或双引号，表示它可能是一个字符串
        if "'" in token or '"' in token: 
            # 如果token在token2mask字典中不存在
            if token2mask.get(token) == None:
                # 生成一个mask，格式为<_str%d_>
                mask = "<_str%d_>"%cnt
                # 将token映射到mask
                token2mask[token] = mask
                # 将mask映射到token
                mask2token[mask] = token
                # 计数器加1
                cnt += 1
            # 将token的mask添加到结果列表中
            result.append(token2mask[token])
        else:
            # 如果token不是字符串，则直接添加到结果列表中
            result.append(token)
    # 返回处理后的tokens列表、mask到token的映射字典和token到mask的映射字典
    return result, token2mask, mask2token

def recoverStr(tokens, mask2token):
    result = []
    for token in tokens:
        if token.startswith("<_str"):
            result.append(mask2token[token])
        else:
            result.append(token)
    return result

def _go4next(tokens, token, curIdx):
    n = len(tokens)
    while curIdx < n and tokens[curIdx] != token:
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx 

def _go4match(tokens, startToken, curIdx):
    endToken = ""
    if startToken == "(":
        endToken = ")"
    elif startToken == "[":
        endToken = "]"
    elif startToken == "{":
        endToken = "}"
    else:
        assert False

    indent = 0
    n = len(tokens)
    while curIdx < n:
        if tokens[curIdx] == startToken:
            indent += 1
        elif tokens[curIdx] == endToken:
            indent -= 1
            if indent == 0:
                break
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx

def _tokens2stmts(tokens, level=0):
    le_paren = 0
    idx = 0
    n = len(tokens)
    res = ""
    res += INDENT * level
    inAssign = False
    while idx < n:
        t = tokens[idx]
        res += t + " "
        if t == "(":
            le_paren += 1
        elif t == ")":
            le_paren -= 1
            if le_paren == 0:               # in case of "if ((a=b)!=0)"
                inAssign = False            
        elif t == ";" and le_paren == 0:    # in case of ";" in "for (int i=0; i<10: i++)"
            res += "\n"
            if idx != n - 1:
                res += INDENT * level
            inAssign = False
        elif t in [";", ",", ":", "?"]:
            inAssign = False
        elif t == "{" and not inAssign:     # in case of "int a [ 10 ] [ 10 ] = { { 0 }, { 0 } };"
            startIdx = idx + 1
            endIdx = _go4match(tokens, "{", idx)
            res += "\n"
            res += _tokens2stmts(tokens[startIdx: endIdx], level + 1)
            res += "\n"
            if endIdx+1 != n:
                res += INDENT * level
            idx = endIdx
        elif t == "{" and inAssign:
            idx += 1
            while idx < n:
                res += tokens[idx] + " "
                if tokens[idx] == ";":
                    res += "\n"
                    if idx != n - 1:
                        res += INDENT * level
                    inAssign = False
                    break
                idx += 1
        elif t in ["=", "enum"]:
            inAssign = True;
        idx += 1
    return res

def _getIndent(str_):
    res = ""
    for ch in str_:
        if ch in ["\t", " "]:
            res += ch
        else:
            break
    return res



# return: (stmts, StmtInsPos)
#   stmts: a list of statements, where variable declaration can be inserted following
# 返回两个参数：
# stmts: 处理后的语句列表
# StmtInsPos: 可以插入语句的位置列表
def tokens2stmts(tokens):
    # 提取字符串
    tokens, token2mask, mask2token = extractStr(tokens)
    # 将tokens转换为语句列表
    tokens = _tokens2stmts(tokens)
    # 将语句列表分割为单独的语句
    stmts = tokens.split("\n")
    # 去除空白语句
    stmts = ["" if stmt.strip() == "" else stmt for stmt in stmts]

    # 添加临时省略的"}"，并保持相应的缩进
    statStack = []
    newStmts = []
    for stmt in stmts:
        if stmt != "":
            newStmts.append(stmt)
            if stmt.rstrip().endswith("{"):
                statStack.append(stmt)
        elif len(statStack) != 0:
            matchStmt = statStack.pop()
            newStmts.append(_getIndent(matchStmt)+"}")
        else:
            pass

    # 匹配结构体、联合体、枚举体或类型定义的开始
    # 定义变量标识符的正则表达式模式
    pattern_uid = "[A-Za-z_][A-Za-z0-9_]*(\[[0-9]*\])*"
    # 定义完整的变量声明模式
    pattern = "^({},\s*)*{};".format(pattern_uid, pattern_uid)
    # 初始化语句列表
    stmts = []
    # 初始化代码块栈
    blockStack = []  
    # 标记是否处于结构体或联合体结束位置
    endStruct = False # 包括结构体和联合体
    # 遍历处理每个语句
    for stmt in newStmts:
        # 如果语句以{结尾,表示代码块开始
        if stmt.rstrip().endswith("{"):
            stmts.append(stmt)
            blockStack.append(stmt)
        # 如果语句是单独的}，表示代码块结束
        elif stmt.strip() == "}":
            stmts.append(stmt)
            # 检查是否是结构体、联合体或typedef的结束
            if blockStack[-1].lstrip().startswith("struct") or\
              blockStack[-1].lstrip().startswith("union") or\
              blockStack[-1].lstrip().startswith("typedef"):
                endStruct = True
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
            if structStack != [] and paren_n == structStack[-1][1]:
                structStack.pop()
            paren_n -= 1
        # 如果不在结构体内,则当前位置可以插入语句
        if structStack == []:
            StmtInsPos.append(i)

    # 获取每个语句的缩进
    indents = [_getIndent(stmt) for stmt in stmts]
    # 恢复被掩码的字符串
    stmts = [recoverStr(stmt.strip().split(), mask2token) for stmt in stmts]   

    # 返回三个参数：
    # stmts: 处理后的语句列表
    # StmtInsPos: 可以插入语句的位置列表
    # indents: 每个语句的缩进信息列表
    return stmts, StmtInsPos, indents

def StmtInsPos(tokens, strict=True):
    '''
    查找可以插入任何语句的所有可能位置
    '''
    # 将tokens转换为语句列表、语句插入位置列表和缩进列表
    statements, StmtInsPos, _ = tokens2stmts(tokens)
    # 初始化结果列表和计数器
    res = []
    cnt, indent = 0, 0
    # 如果不是严格模式，则在开始位置插入一个可能的插入位置
    if not strict:
        res.append(-1)
    # 遍历每个语句
    for i, stmt in enumerate(statements):
        # 计数器增加当前语句的长度
        cnt += len(stmt)
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
    # 返回所有可能的插入位置
    return res

def DeclInsPos(tokens):
    '''
    查找所有可以插入变量声明的位置
    '''
    # 将tokens转换为语句列表，忽略插入位置和缩进信息
    statements, _, _ = tokens2stmts(tokens)
    # 初始化结果列表和计数器
    res = []
    cnt = 0
    # 在开始位置添加一个插入点(-1表示在第一个语句之前)
    res.append(-1)
    # 遍历每个语句
    for stmt in statements:
        # 累加语句长度
        cnt += len(stmt)
        # 在每个语句后添加一个插入点
        res.append(cnt-1)
    # 返回所有可能的插入位置
    return res

def BrchInsPos(tokens):
    '''
    Find all possible positions to insert false branch that control flow will never reach
    '''
    return StmtInsPos(tokens)

def LoopInsPos(tokens):
    '''
    Find all possible positions to insert loop that has no effect
    '''
    return StmtInsPos(tokens)

def FuncInsPos(tokens):
    '''
    Find all possible positions to insert functions
    '''
    return StmtInsPos(tokens)

def _InsVis(tokens, pos):
    statements, _, indents = tokens2stmts(tokens)
    lens = [len(line) for line in statements]

    for pidx in pos:
        if pidx == -1:
            statements[0] = ["[____]"] + statements[0]
            continue
        cnt = 0
        for i, n in enumerate(lens):
            cnt += n
            if cnt > pidx:
                statements[i].append("[____]")
                break

    for indent, line in zip(indents, statements):
        print(indent, end="")
        print(" ".join(line))



# return [pos1, pos2, ...]
def InsAddCandidates(insertDict, maxLen=None):

    res = []
    for pos in insertDict.keys():
        if pos == "count":
            continue
        if maxLen is None:
            res.append(pos)
        elif int(pos) < maxLen:
            res.append(pos)
    return res

# only able to insert into legal poses, and can't insert same inserted content into same pos (return False)
def InsAdd(insertDict, pos, insertedTokenList):

    suc = True
    assert insertDict.get(pos) is not None  # this position could be inserted
    if insertedTokenList in insertDict[pos]:    # can't insert same statement
        suc = False
    else:
        insertDict[pos].append(insertedTokenList)

    if suc:
        if insertDict.get("count") is not None:
            insertDict["count"] += 1
        else:
            insertDict["count"] = 1
    return suc

# return [(pos1, 0), (pos1, 1), (pos2, 0), ...]
def InsDeleteCandidates(insertDict):
    """
    获取所有可以删除的候选位置。
    
    参数:
    - insertDict: 一个字典，包含了插入位置和对应的插入内容列表。
    
    返回:
    - res: 一个列表，包含了所有可以删除的候选位置，每个位置是一个元组，包含了插入位置和列表索引。
    """
    res = []
    # 遍历insertDict字典中的所有键
    for key in insertDict.keys():
        # 如果键是"count"，则跳过
        if key == "count":
            continue
        # 如果键对应的值不是空列表，则遍历列表
        if insertDict[key] != []:
            for i, _ in enumerate(insertDict[key]):
                # 将键和列表索引作为元组添加到结果列表中
                res.append((key, i))
    return res

# what is passed in must is legal (pos, listIdx)
def InsDelete(insertDict, pos, listIdx=0):

    assert insertDict.get(pos) is not None
    assert insertDict[pos] != []
    if len(insertDict[pos]) <= listIdx:
        assert False
    del insertDict[pos][listIdx]
    assert insertDict.get("count") is not None
    insertDict["count"] -= 1

# return complete tokenlist by inserting corresponding token list (insertions)
def InsResult(tokens, insertDict):

    result = []
    if insertDict.get(-1) is not None:
        for tokenList in insertDict[-1]:
            result += tokenList
    for i, t in enumerate(tokens):
        result.append(t)
        if insertDict.get(i) is not None:   # so it's a legal insertion position
            for tokenList in insertDict[i]:
                result += tokenList
    return result



# Usage: for each token in <tokens>, find "end token" that indicating where the corresponding statement ends.
#   tokens: token list
#   optimize: optimize level
#       0 => Label index of statement end token within single line.                         E.g. "for(...) sum++;" => index of ";"
#       1 => Label index of recurrent statement block end token (including "switch").       E.g. "while() {}"、"do {} while ();"、"for() {}"
#       2 => Label index of if-else statement block end token.                              E.g. "if() {} else if() {} else xxx;" => ";"
#       3 => Label index of other open statement "block" end token.                         E.g. "int() {}" => "}"       "enum { 1, 5, a }" => "}"
# Return: <list<int>> of index of end tokens
def getStmtEnd(tokens, optimize=3):

    statements, _, indents = tokens2stmts(tokens)
    heads = [stmt[0] for stmt in statements]
    ends = [stmt[-1] for stmt in statements]
    lens = [len(stmt) for stmt in statements]
    n = len(ends)
    endIndices = []

    # end token index for each line (single line statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if end == ";":
            if i == n-1:
                endIndices.append(totalCnt - 1)
            elif head not in ["for", "while", "do", "if", "else", "switch"]:
                endIndices.append(totalCnt - 1)
            elif head in ["for", "while", "switch", "do"]:
                endIndices.append(totalCnt - 1)
            elif heads[i+1] != "else":
                endIndices.append(totalCnt - 1)
            else:
                endIndices.append(None)
        elif end == "}":
            if i == n-1:
                endIndices.append(totalCnt - 1)
            elif len(indents[i+1]) < len(indent):
                endIndices.append(totalCnt - 1)
            elif heads[i+1] != "else" and heads[i+1] != "while":
                endIndices.append(totalCnt - 1)
            else:
                endIndices.append(None)
        else:
            endIndices.append(None)
    if optimize <= 0:
        return endIndices

    # end token index for each line ("for { }" & "switch { }" & "while { }" & "do { } while ();" block statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["for", "while", "switch"]:
            continue
        if end == "{":
            curIdx = i + 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["do"]:
            continue
        if end == "{":
            curIdx = i + 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            curIdx += 1
            while curIdx < n and not (indents[curIdx]==indent and heads[curIdx]=="while" and ends[curIdx]==";"):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "while", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    if optimize <= 1:
        return endIndices

    # end token index for each line ("if else" statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["if", "else"]:
            continue
        curTotalCnt = totalCnt
        curIdx = i
        while True:
            curIdx += 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curTotalCnt += lens[curIdx]
                curIdx += 1
            assert curIdx < n   # because all single if/else statements have been processed in o-0
            if endIndices[curIdx] != None:
                endIndices[i] = endIndices[curIdx]
                break
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if not (head=="}" and i+1<n and heads[i+1]=="else"):
            continue
        endIndices[i] = endIndices[i+1]
    if optimize <= 2:
        return endIndices

    # end token index for each line (left "{ }" block statement, e.g. "int main() {}" & "enum { ...; }")
    # WARNING! This WILL occur to assertion error. NO GUARANTEE!
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if end == "{":
            curIdx = i + 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    
    res = []
    for cnt, endIdx in zip(lens, endIndices):
        res += [endIdx] * cnt
    return res

def IfElseReplacePos(tokens, endPoses):
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "if":     # only "if {} else {}", ont process "else if {}"
            ifIdx = i
            conditionEndIdx = _go4match(tokens, "(", ifIdx)
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = _go4match(tokens, "{", conditionEndIdx + 1)
            else:
                ifBlockEndIdx = _go4next(tokens, ";", conditionEndIdx + 1)
            if not (ifBlockEndIdx + 1 < n and tokens[ifBlockEndIdx + 1] == "else"):
                continue
            if tokens[ifBlockEndIdx + 2] == "if":   # in case of "else if {}"
                continue
            elseBlockEndIdx = endPoses[ifBlockEndIdx + 1]
            pos.append([ifIdx, conditionEndIdx, ifBlockEndIdx, elseBlockEndIdx])
    return pos

def IfElseReplace(tokens, pos):
    beforeIf = tokens[:pos[0]]
    codition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        blockIf = tokens[pos[1]+2:pos[2]]
    else:
        blockIf = tokens[pos[1]+1:pos[2]+1]
    if tokens[pos[2]+2] == "{":
        blockElse = tokens[pos[2]+3:pos[3]]
    else:
        blockElse = tokens[pos[2]+2:pos[3]+1]
    afterElse = tokens[pos[3]+1:]
    res = beforeIf + ["if", "(", "!", "("] + codition + [")", ")", "{"] + blockElse + ["}", "else", "{"] + blockIf + ["}"] + afterElse
    return res 

def IfReplacePos(tokens, endPoses):
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "if":
            ifIdx = i
            conditionEndIdx = _go4match(tokens, "(", ifIdx)
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = _go4match(tokens, "{", conditionEndIdx + 1)
            else:
                ifBlockEndIdx = _go4next(tokens, ";", conditionEndIdx + 1)
            if ifBlockEndIdx + 1 < n and tokens[ifBlockEndIdx + 1] == "else":   # in case of "if {} else {}", only process "if {} xxx"
                continue
            pos.append([ifIdx, conditionEndIdx, ifBlockEndIdx])
    return pos

def IfReplace(tokens, pos):
    beforeIf = tokens[:pos[0]]
    condition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        body = tokens[pos[1]+2:pos[2]]
    else:
        body = tokens[pos[1]+1:pos[2]+1]
    afterIf = tokens[pos[2]+1:]
    # if (a) {} => if (!a); else {}
    res = beforeIf + ["if", "(", "!", "("] + condition + [")", ")", ";", "else", "{"] + body + ["}"] + afterIf
    return res 

def For2WhileReplacePos(tokens, endPoses):
    pos = []
    for i, t in enumerate(tokens):
        if t == "for":
            forIdx = i
            conditionEndIdx = _go4match(tokens, "(", forIdx)
            if tokens[conditionEndIdx + 1] == "{":
                blockForEndIdx = _go4match(tokens, "{", conditionEndIdx)
            else:
                blockForEndIdx = endPoses[conditionEndIdx + 1]
            condition1EndIdx = _go4next(tokens, ";", forIdx)
            condition2EndIdx = _go4next(tokens, ";", condition1EndIdx + 1)
            pos.append([forIdx, condition1EndIdx, condition2EndIdx, conditionEndIdx, blockForEndIdx])
    return pos

def For2WhileRepalce(tokens, pos):
    beforeFor = tokens[:pos[0]]
    condition1 = tokens[pos[0]+2:pos[1]+1]
    condition2 = tokens[pos[1]+1:pos[2]]
    condition3 = tokens[pos[2]+1:pos[3]] + [";"]
    if tokens[pos[3]+1] == "{":
        body = tokens[pos[3]+2:pos[4]]
    else:
        body = tokens[pos[3]+1:pos[4]+1]
    afterFor = tokens[pos[4]+1:]
    if beforeFor != [] and beforeFor[-1] in [";", "{", "}"]:
        res = beforeFor + condition1 + ["while", "("] + condition2 + [")", "{"] + body + condition3 + ["}"] + afterFor
    else:
        res = beforeFor + ["{"] + condition1 + ["while", "("] + condition2 + [")", "{"] + body + condition3 + ["}", "}"] + afterFor
    return res    

def While2ForReplacePos(tokens, endPoses):
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "while":
            whileIdx = i
            conditionEndIdx = _go4match(tokens, "(", whileIdx)
            if conditionEndIdx + 1 < n and tokens[conditionEndIdx + 1] == ";":   # in case of "do {} while ();"
                continue
            if tokens[conditionEndIdx + 1] == "{":
                blockWhileEndIdx = _go4match(tokens, "{", conditionEndIdx)
            else:
                blockWhileEndIdx = endPoses[conditionEndIdx + 1]
            pos.append([whileIdx, conditionEndIdx, blockWhileEndIdx])
    return pos

def While2ForRepalce(tokens, pos):
    beforeWhile = tokens[:pos[0]]
    condition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        body = tokens[pos[1]+2:pos[2]]
    else:
        body = tokens[pos[1]+1:pos[2]+1]
    afterWhile = tokens[pos[2]+1:]
    res = beforeWhile + ["for", "(", ";"] + condition + [";", ")", "{"] + body + ["}"] + afterWhile
    return res 



if __name__ == "__main__":
    
    with gzip.open('../data/oj.pkl.gz', "rb") as f:
        d = pickle.load(f)
        raw = d['raw_tr'] + d['raw_te']

    '''for i, code in enumerate(raw):
        if "struct" in code:
            print("%d-----------------------------------------"%i)
            pos = StmtInsPos(code) #pos = StmtInsPos(code)
            _InsVis(code, pos)
            print("%d~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"%i)
            stmts, _, indents = tokens2stmts(code)
            endTokenIndices = getStmtEnd(code, 3)
            for stmt, endIdx, indent in zip(stmts, endTokenIndices, indents):
                if endIdx != None:
                    print("{:30}".format(" ".join(code[endIdx-3:endIdx+1])), end="")
                else:
                    print("{:30}".format(" "), end="")
                print(indent + " ".join(stmt))'''

    import random


    # TEST INSERT & DELETE
    import pycparser
    parser = pycparser.c_parser.CParser()
    inserts = [
        ";",
        "if ( false ) ;",
        "if ( true ) ;",
        "if ( false ) ; else ;"]
    inserts = [insert.split(" ") for insert in inserts]
    for _ in range(1000):
        code = random.sample(raw, 1)[0]
        poses = StmtInsPos(code, strict=True) # True to make sure that parse succeed (insert to position within block)
        insertDict = dict([(pos, []) for pos in poses])
        _InsVis(code, poses)
        print("-----------------------------------")
        suc_cnt = 0
        for _ in range(10):
            candis = InsAddCandidates(insertDict)
            insIdx = random.randint(0, len(candis)-1)
            pos = candis[insIdx]
            instIdx = random.randint(0, len(inserts)-1)
            inst = inserts[instIdx]
            if InsAdd(insertDict, pos, inst):
                suc_cnt += 1
            _InsVis(InsResult(code, insertDict), [])
            parser.parse(" ".join(InsResult(code, insertDict)))
            print("------------- INSERT ---------------", insertDict["count"])
        for _ in range(suc_cnt): 
            candis = InsDeleteCandidates(insertDict)
            delIdx = random.randint(0, len(candis)-1)
            pos, listIdx = candis[delIdx]
            InsDelete(insertDict, pos, listIdx)
            _InsVis(InsResult(code, insertDict), [])
            parser.parse(" ".join(InsResult(code, insertDict)))
            print("------------- REMOVE --------------", insertDict["count"])
        print()

    # TEST FOR => WHILE
    '''code = raw[2333]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = For2WhileReplacePos(code, endPoses)
    for pos in poses:
        res = For2WhileRepalce(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    # TEST WHILE => FOR
    '''code = raw[233]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = While2ForReplacePos(code, endPoses)
    for pos in poses:
        res = While2ForRepalce(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    # TEST FINAL IF ELSE
    '''code = raw[233]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = IfElseReplacePos(code, endPoses)
    for pos in poses:
        res = IfElseReplace(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    # TEST SINGLE IF
    '''code = raw[233]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = IfReplacePos(code, endPoses)
    for pos in poses:
        res = IfReplace(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    