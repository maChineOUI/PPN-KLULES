cmake_minimum_required(VERSION 3.16)

project(lulesh_kokkos LANGUAGES CXX)

# --- Configuration du standard C++ requis par Kokkos 5.0
# --- Kokkos 5.0 要求的 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Localisation de Kokkos installé sur votre machine
# --- 指定你的 Kokkos 安装位置
#     Remplacez /Users/wei/kokkos-install par votre chemin réel
#     请把 /Users/wei/kokkos-install 替换为你真实的 Kokkos 安装路径
set(Kokkos_ROOT "$ENV{HOME}/kokkos-install")

# --- Trouver la bibliothèque Kokkos (CMake find_package)
# --- 使用 CMake find_package 方式找到 Kokkos
find_package(Kokkos REQUIRED)

# --- Rassembler tous les fichiers .cc de la racine du projet
# --- 自动收集项目根目录中的所有 .cc 文件
file(GLOB LULESH_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

# --- Création de l'exécutable
# --- 创建可执行文件
add_executable(lulesh.host
    ${LULESH_SOURCES}
)

# --- Répertoires d'inclusion
# --- 头文件路径
target_include_directories(lulesh.host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
)

# --- Définitions équivalentes à celles du Makefile ancienne version
# --- 与旧 Makefile 中一致的宏定义
target_compile_definitions(lulesh.host PRIVATE
    USE_MPI=0
)

# --- Lier Kokkos à l'exécutable
# --- 链接 Kokkos 库
target_link_libraries(lulesh.host PRIVATE Kokkos::kokkos dl)
