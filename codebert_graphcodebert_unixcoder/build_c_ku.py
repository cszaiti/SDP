# build_c_parser.py
from tree_sitter import Language
import os
import shutil

def build_language_lib():
    """构建 tree-sitter C语言解析器"""
    
    # 1. 创建临时目录
    os.makedirs('vendor', exist_ok=True)
    
    # 2. 克隆 tree-sitter-c 仓库
    if not os.path.exists('vendor/tree-sitter-c'):
        clone_cmd = 'git clone https://github.com/tree-sitter/tree-sitter-c vendor/tree-sitter-c'
        if os.system(clone_cmd) != 0:
            raise Exception("Failed to clone tree-sitter-c repository")
    
    # 3. 构建语言库
    os.makedirs('build', exist_ok=True)
    lib_path = os.path.abspath('build/c-language.so')
    
    try:
        Language.build_library(
            lib_path,
            ['vendor/tree-sitter-c']
        )
        print(f"Successfully built language library at: {lib_path}")
        
        # 4. 复制到项目目录
        os.makedirs('libs', exist_ok=True)
        shutil.copy(lib_path, 'libs/c.so')
        print(f"Copied to libs/c.so")
        
    except Exception as e:
        print(f"Error building language library: {e}")
        return False
    
    # 5. 清理临时文件（可选）
    shutil.rmtree('vendor')
    shutil.rmtree('build')
    
    return True

if __name__ == "__main__":
    build_language_lib()