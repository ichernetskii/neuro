#!/bin/bash

input_dir="./mnist_png"
output_dir="./mnist_png_bw"

# Создаем выходную директорию, если её нет
mkdir -p "$output_dir"

# Проходим по всем PNG-файлам
find "$input_dir" -type f -name "*.png" | while read -r file; do
  # Получаем относительный путь
  rel_path="${file#$input_dir/}"
  # Создаем путь к выходному файлу
  output_file="$output_dir/$rel_path"
  # Создаем директорию, если её нет
  mkdir -p "$(dirname "$output_file")"

  # Выполняем преобразование
  magick "$file" -colorspace RGB -color-threshold 'rgb(128,128,128)-rgb(255,255,255)' "$output_file"
done
