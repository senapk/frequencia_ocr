#! /bin/bash
 
 # Extrai imagens de um PDF usando o pdftoppm
 # Requer que o poppler-utils esteja instalado (sudo apt install poppler-utils)
 # archlinux
 # sudo pacman -S poppler
 
 if [ "$#" -ne 2 ]; then
     echo "Uso: $0 arquivo.pdf prefixo_saida"
     exit 1
 fi
 
 INPUT_PDF="$1"
 PREFIX_OUTPUT="$2"
 
 pdftoppm -png "$INPUT_PDF" "${PREFIX_OUTPUT}"
 
 echo "Imagens extraídas para ${PREFIX_OUTPUT}_page-1.png, ${PREFIX_OUTPUT}_page-2.png, etc."