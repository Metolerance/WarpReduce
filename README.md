# WarpReduce

一个高效的CUDA warp级别归约操作实现项目，展示了GPU并行计算中的归约算法优化技术。

## 项目简介

本项目实现了两种不同的warp级别归约算法：
- **标准warp归约**：基础的warp内归约实现
- **带填充的warp归约**：优化版本，通过填充技术提高性能

## 功能特性

- 高性能GPU并行归约计算
- 支持自定义归约操作（如求和、求最大值等）
- 模板化设计，支持多种数据类型
- 性能分析和基准测试
- 完整的CUDA错误检查机制

## 项目结构

```
WarpReduce/
├── src/                          # 源代码目录
│   ├── main.cu                   # 主程序入口
│   ├── warp_reduce.cu            # 标准warp归约实现
│   └── warp_reduce_with_padding.cu # 带填充的warp归约实现
├── include/                      # 头文件目录
│   ├── common.cuh                # 通用定义和工具
│   ├── warp_reduce.cuh           # 标准归约函数声明
│   └── warp_reduce_with_padding.cuh # 填充归约函数声明
├── CMakeLists.txt                # CMake构建配置
├── result.csv                    # 性能测试结果
└── README.md                     # 项目说明文档
```

## 环境要求

- **CUDA Toolkit**: 支持CUDA的GPU和驱动
- **CMake**: 3.26或更高版本
- **C++标准**: C++17

## 核心算法

### Warp级别归约
- 利用CUDA的warp内线程同步特性
- 使用`__shfl_down_sync`进行高效数据交换
- 减少共享内存访问，提高性能

### 块级别归约
- 多个warp结果的进一步归约
- 支持任意大小的数据集
- 自动处理边界条件

## 性能优化

项目包含两种实现方式的性能对比：
- 标准实现：基础的归约算法
- 填充优化：通过数据填充减少分支分歧，提高执行效率



