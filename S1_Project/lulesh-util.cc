#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#if USE_MPI
#include <mpi.h>
#endif
#include "lulesh.h"

/* 将字符串转换为整数（带错误检查）
   Conversion d'une chaîne vers entier avec vérification d'erreurs */
int StrToInt(const char *token, int *retVal)
{
  const char *c;
  char *endptr;
  const int decimal_base = 10;

  if (token == NULL)
    return 0;

  c = token;
  *retVal = (int)strtol(c, &endptr, decimal_base);

  if ((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
    return 1;
  else
    return 0;
}

/* 打印命令行选项
   Affiche les options de la ligne de commande */
static void PrintCommandLineOptions(char *execname, int myRank)
{
  if (myRank == 0) {

    printf("Usage: %s [opts]\n", execname);
    // 使用方式：程序名 + 选项
    // Mode d'emploi : nom du programme + options

    printf(" where [opts] is one or more of:\n");
    // 可用选项如下
    // Les options disponibles sont :

    printf(" -q              : quiet mode - suppress all stdout\n");
    // 静默模式：关闭所有输出
    // Mode silencieux : supprime toutes les sorties

    printf(" -i <iterations> : number of cycles to run\n");
    // 指定迭代次数
    // Nombre d’itérations à exécuter

    printf(" -s <size>       : length of cube mesh along side\n");
    // 网格边长
    // Taille d’un côté du maillage cubique

    printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
    // 区域数量（默认 11）
    // Nombre de régions distinctes (défaut : 11)

    printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
    // 区域负载均衡参数
    // Facteur d’équilibrage de charge entre régions

    printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
    // 高代价区域的额外开销
    // Coût supplémentaire pour régions plus coûteuses

    printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
    // 可视化文件分片数量
    // Nombre de fichiers pour diviser la sortie de visualisation

    printf(" -p              : Print out progress\n");
    // 输出运行进度
    // Affiche la progression

    printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
    // 输出可视化文件（需 -DVIZ_MESH）
    // Génère un fichier de visualisation (nécessite -DVIZ_MESH)

    printf(" -h              : This message\n");
    // 显示帮助信息
    // Affiche ce message d’aide

    printf("\n\n");
  }
}

/* 打印解析错误并退出
   Affiche une erreur d'analyse et quitte */
static void ParseError(const char *message, int myRank)
{
  if (myRank == 0) {
    printf("%s\n", message);
    // 输出错误信息
    // Affiche le message d’erreur

#if USE_MPI
    MPI_Abort(MPI_COMM_WORLD, -1);
#else
    exit(-1);
#endif
  }
}

/* 解析命令行参数
   Analyse les options de la ligne de commande */
void ParseCommandLineOptions(int argc, char *argv[],
                             int myRank, struct cmdLineOpts *opts)
{
  if (argc > 1) {

    int i = 1;

    while (i < argc) {

      int ok;

      /* -i <iterations> */
      if (strcmp(argv[i], "-i") == 0) {

        if (i + 1 >= argc) {
          ParseError("Missing integer argument to -i", myRank);
          // -i 缺少参数
          // Argument manquant pour -i
        }

        ok = StrToInt(argv[i + 1], &(opts->its));

        if (!ok) {
          ParseError("Parse Error on option -i integer value required after argument\n", myRank);
          // -i 后必须接整数
          // -i requiert une valeur entière
        }

        i += 2;
      }

      /* -s <size> */
      else if (strcmp(argv[i], "-s") == 0) {

        if (i + 1 >= argc) {
          ParseError("Missing integer argument to -s\n", myRank);
          // -s 缺少参数
          // Argument manquant pour -s
        }

        ok = StrToInt(argv[i + 1], &(opts->nx));

        if (!ok) {
          ParseError("Parse Error on option -s integer value required after argument\n", myRank);
          // -s 后必须接整数
          // -s requiert une valeur entière
        }

        i += 2;
      }

      /* -r <numregions> */
      else if (strcmp(argv[i], "-r") == 0) {

        if (i + 1 >= argc) {
          ParseError("Missing integer argument to -r\n", myRank);
          // -r 缺少参数
          // Argument manquant pour -r
        }

        ok = StrToInt(argv[i + 1], &(opts->numReg));

        if (!ok) {
          ParseError("Parse Error on option -r integer value required after argument\n", myRank);
          // -r 后必须接整数
          // -r requiert une valeur entière
        }

        i += 2;
      }

      /* -f <numfilepieces> */
      else if (strcmp(argv[i], "-f") == 0) {

        if (i + 1 >= argc) {
          ParseError("Missing integer argument to -f\n", myRank);
          // -f 缺少参数
          // Argument manquant pour -f
        }

        ok = StrToInt(argv[i + 1], &(opts->numFiles));

        if (!ok) {
          ParseError("Parse Error on option -f integer value required after argument\n", myRank);
          // -f 后必须接整数
          // -f requiert une valeur entière
        }

        i += 2;
      }

      /* -p */
      else if (strcmp(argv[i], "-p") == 0) {
        opts->showProg = 1;
        // 记录输出进度标志
        // Active l’affichage de progression
        i++;
      }

      /* -q */
      else if (strcmp(argv[i], "-q") == 0) {
        opts->quiet = 1;
        // 记录静默模式
        // Active le mode silencieux
        i++;
      }

      /* -b <balance> */
      else if (strcmp(argv[i], "-b") == 0) {

        if (i + 1 >= argc) {
          ParseError("Missing integer argument to -b\n", myRank);
          // -b 缺少参数
          // Argument manquant pour -b
        }

        ok = StrToInt(argv[i + 1], &(opts->balance));

        if (!ok) {
          ParseError("Parse Error on option -b integer value required after argument\n", myRank);
          // -b 后必须接整数
          // -b requiert une valeur entière
        }

        i += 2;
      }

      /* -c <cost> */
      else if (strcmp(argv[i], "-c") == 0) {

        if (i + 1 >= argc) {
          ParseError("Missing integer argument to -c\n", myRank);
          // -c 缺少参数
          // Argument manquant pour -c
        }

        ok = StrToInt(argv[i + 1], &(opts->cost));

        if (!ok) {
          ParseError("Parse Error on option -c integer value required after argument\n", myRank);
          // -c 后必须接整数
          // -c requiert une valeur entière
        }

        i += 2;
      }

      /* -v */
      else if (strcmp(argv[i], "-v") == 0) {

#if VIZ_MESH
        opts->viz = 1;
        // 记录输出可视化标志
        // Active la sortie de visualisation
#else
        ParseError("Use of -v requires compiling with -DVIZ_MESH\n", myRank);
        // 未启用可视化编译选项
        // -v nécessite -DVIZ_MESH
#endif
        i++;
      }

      /* -h */
      else if (strcmp(argv[i], "-h") == 0) {

        PrintCommandLineOptions(argv[0], myRank);
        // 显示帮助信息
        // Affiche l’aide

#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 0);
#else
        exit(0);
#endif
      }

      /* 未知选项 */
      else {
        char msg[80];
        PrintCommandLineOptions(argv[0], myRank);
        sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
        // 未知命令行参数
        // Option de ligne de commande inconnue
        ParseError(msg, myRank);
      }
    }
  }
}
