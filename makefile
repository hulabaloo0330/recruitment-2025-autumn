# 指定 nvcc 的完整路径
CXX = /usr/local/cuda/bin/nvcc

# 为您的GPU架构设置正确的标志以获得最佳性能
# sm_89: Ada Lovelace (e.g., RTX 4080)
GPU_ARCH = -gencode=arch=compute_89,code=sm_89

# 为 nvcc 设置C++14标准
# -I./inc 使得编译器可以找到 gmres.hpp 和 sparseMatrix.hpp
NVCC_CFLAGS = -std=c++14 -O3 $(GPU_ARCH) -I./inc  

# 链接 CUDA 库
CXX_LDFLAGS = -lcublas -lcusparse

# 目录
DIR_OBJ = obj
DIR_SRC = src
DIR_INC = inc

# 列出所有的源文件
CPP_SRCS = $(DIR_SRC)/main.cpp $(DIR_SRC)/gmres.cpp

# 自动生成目标文件名列表 (e.g., obj/main.o obj/gmres.o)
AOBJS = $(patsubst $(DIR_SRC)/%.cpp, $(DIR_OBJ)/%.o, $(CPP_SRCS))

# 最终可执行文件名
PROG = gmres

# --- 编译和链接规则 ---

# 默认目标
all: $(PROG)

# 链接规则: 将所有 .o 文件链接成可执行文件
$(PROG): $(AOBJS)
	$(CXX) $(NVCC_CFLAGS) $^ -o $@ $(CXX_LDFLAGS)

# 编译规则: 将每个 .cpp 文件编译成 .o 文件
$(DIR_OBJ)/%.o: $(DIR_SRC)/%.cpp
	@mkdir -p $(DIR_OBJ)
	$(CXX) -c $(NVCC_CFLAGS) $< -o $@

# --- 清理规则 ---

.PHONY : clean
clean :
	rm -f $(DIR_OBJ)/*.o $(PROG)