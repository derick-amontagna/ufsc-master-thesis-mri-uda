#!/bin/bash
data_dir=data/OfficeHome

# ERM
config_path=tests/transfer_loss/ERM.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart


# BNM
config_path=tests/transfer_loss/BNM.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart


# DAAN
config_path=tests/transfer_loss/DAAN.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart


# DAN
config_path=tests/transfer_loss/DAN.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart


# DANN
config_path=tests/transfer_loss/DANN.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart


# DeepCoral
config_path=tests/transfer_loss/DeepCoral.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart


# DSAN
config_path=tests/transfer_loss/DSAN.yaml
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain Art --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Clipart --tgt_domain Product

python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain RealWorld
python main.py --config $config_path --data_dir $data_dir --src_domain Product --tgt_domain Clipart

python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Art
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Product
python main.py --config $config_path --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart