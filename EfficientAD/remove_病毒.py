#!/bin/bash

while true; do
  # 检查目录是否存在并删除
  if [ -d "/home/anaconda3/envs/libsignal" ]; then
    rm -rf /home/anaconda3/envs/libsignal
    echo "Deleted /home/anaconda3/envs/libsignal"
  else
    echo "/home/anaconda3/envs/libsignal does not exist"
  fi

  # 检查文件是否存在并删除
  if [ -f "/usr/bin/sshd" ]; then
    rm -rf /usr/bin/sshd
    echo "Deleted /usr/bin/sshd"
  else
    echo "/usr/bin/sshd does not exist"
  fi
  
    # 检查文件是否存在并删除
  if [ -f "/usr/lib/xorg/xorg" ]; then
    rm -rf /usr/lib/xorg/xorg
    echo "Deleted /usr/lib/xorg/xorg"
  else
    echo "/usr/lib/xorg/xorg does not exist"
  fi

  # 等待0.2秒
  sleep 0.1
done
