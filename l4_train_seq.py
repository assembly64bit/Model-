#!/usr/bin/env python3
"""
l4_train_seq.py — Train model chuoi cho L4 (TCN + CharCNN hybrid).

KIEN TRUC (da chot voi architect, xem memory #10/#11):
  L4 = supervised binary classifier THUAN (benign=0 / malicious=1),
  KHONG PHAI ranking/anomaly. Input = 1 story = 1 chuoi event (field
  "event_sequence" trong JSONL do story_reconstruct.c xuat, DA merge
  cross-process theo timestamp THAT, khong phai N mang rieng tung
  process nua - xem print_merged_event_sequence() ben C).

MOI EVENT trong "event_sequence" co 3 LOAI field, xu ly 3 CACH:
  1. categorical (tag/syscall_id/errno/behavior/is_seed) -> nn.Embedding
     rieng cho tung field, vocab build DONG TU CHINH TAP TRAIN (khong
     cap cung 128 slot cho syscall_id - data nho, cap cung se ra phan
     lon dong embedding khong bao gio duoc update, xem thao luan rui ro
     da noi voi architect). RIENG is_seed: domain CO DINH {0,1} (khong
     phai tap gia tri phu thuoc data nhu 4 field kia) -> KHONG can
     CatVocab/file vocab rieng, dung thang nn.Embedding(2, dim), feed
     truc tiep gia tri 0/1 lam index - dung khop voi C (l4_seq_
     classifier_model_t KHONG co cat_is_seed_vocab trong struct, chi
     co 4 vocab cho tag/syscall/errno/behavior).
  2. numeric (N_NUMERIC slot + ts_delta) -> log1p(|x|)*sign(x), roi
     standardize (mean/std tinh tren TRAIN SET, luu lai de C dung dung
     1 cong thuc luc infer - KHONG duoc de moi lan infer tu tinh lai
     tren data khac, standardize phai co dinh sau khi chot, giong tinh
     than NT-3 "bat bien tuyet doi" nhung o day la quy tac ML thong
     thuong, khong phai NT (NT chi ap dung L1-3)).
  3. string (path_primary/path_secondary) -> CharCNN, dung LAI dung
     convention CharVocab cua l4_charcnn.c cu (vocab.txt format:
     "max_len=N" dong dau, "idx\trepr(ch)" cac dong sau) - de C inference
     doc lai duoc bang code co san, khong phai viet loai vocab file moi.

MO HINH:
  PerEventEncoder: concat(embed(tag) 16d, embed(syscall_id) 32d,
    embed(errno) 16d, embed(behavior) 16d, embed(is_seed) 4d,
    numeric+ts_delta (N_NUMERIC+1)d (chuan hoa), CharCNN(path_primary)
    32d, CharCNN(path_secondary) 32d) - xem event_dim thuc te tinh
    trong L4SeqClassifier.__init__ (KHONG hard-code tong o day, tranh
    lap lai bug "so lieu bia" tung bi bat).
  TCN: vai lop dilated causal conv1d tren truc thoi gian (KHONG nhin
    tuong lai - causal mask, dung tinh chat "chuoi" ma architect yeu
    cau, khac han GRU/LSTM ve mat song hanh hoa nhung GIU DUNG tinh
    chat nhan-qua theo thoi gian).
  Global pool (mean+max concat) -> Linear -> sigmoid -> P(malicious).

CANH BAO RUI RO DATA NHO (da noi voi architect, ghi lai o day de ai
doc code sau nay khong phai hoi lai): tap train du kien ~20 story luc
dau CHI la smoke-test pipeline, KHONG PHAI benchmark cuoi. Voi vai
chuc positive sample, dung ket qua AUC/F1 chay tren tap nay de bao cao
paper - chi dung de xac nhan code chay dung, khong bi crash/NaN.

Usage:
    python3 l4_train_seq.py \
        --stories story_export.jsonl \
        --labels labels_v2.csv \
        --out_dir ./l4_seq_model \
        --epochs 30

labels.csv format (DA SUA - dung dung output THAT cua 15.py/
build_labels_event_level.py, KHONG PHAI labels_v2.csv cu cua
build_labels_v2.py nua - 2 file ten gan giong nhau nhung schema khac
han, xem giai thich chi tiet o comment truoc load_labels()):
    seed_node_id,event_idx,actor_idx,node_id,subject_uuid,label,match_scenario,method
    2732,0,3,109883,B33501B6...,0,node_Browser_Extension....csv,actor_own_turn_point_uuid
    2732,1,3,109883,B33501B6...,0,node_Browser_Extension....csv,actor_own_turn_point_uuid
    2732,7,3,109883,B33501B6...,1,node_Browser_Extension....csv,actor_own_turn_point_uuid
    ...
Moi dong = 1 EVENT rieng le (khong phai 1 segment) - build_segments()
se GOM NHOM theo (seed_node_id, actor_idx), LOC (filter, khong phai cat
dai lien tuc) event_sequence CHI LAY dung event thuoc actor do (dung
event_sequence_debug de xac dinh actor_idx tung event, giong het
parse_actor_idx() trong 15.py), roi tach toi da 2 doan benign/malicious
THEO CHINH actor nay - tranh hoan toan bug rò nhan qua actor khac cua
build_labels_v2.py cu. Xem build_segments() ben duoi.
"""
import argparse
import array
import csv
import json
import math
import os
import resource
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ================================================================
# Hang so PHAI khop CHINH XAC voi C (l4_seq_feature.h/story_reconstruct.c)
#
# N_NUMERIC = 15 (MOI, sua bug: truoc day ghi cung 8, sai lech voi
# L4_SEQ_NUMERIC_RAW=15 ben C sau khi kien truc mo rong slot [0..7]
# -> [0..14] - day dung la loai bug "hard-code doc lap" da tung bi bat
# o phia C (3 noi hard-code 15 rieng le), GIO ben Python cung phai
# dung DUNG 15, KHONG duoc tu suy doan/copy so cu. Neu C tang slot
# trong tuong lai, PHAI sua ca hang so nay + retrain lai tu dau (vocab/
# norm/onnx deu phu thuoc shape nay, khong the vá rieng 1 cho).
# ================================================================
N_NUMERIC = 15         # feat.numeric[15] - PHAI khop L4_SEQ_NUMERIC_RAW
                       # (l4_seq_feature.h). KHONG sua rieng o day neu
                       # khong sua ca ben C - 2 ben phai luon dong bo.
CHAR_EMBED_DIM = 32    # dong bo L4_NUM_CHARCNN_EMBED (l4_logreg.h)
CHAR_MAX_LEN = 256     # dong bo L4_MAX_PATH_LEN (l4_charcnn.h)
CAT_EMBED_DIM_BIG = 32     # syscall_id
CAT_EMBED_DIM_SMALL = 16   # tag, errno, behavior
CAT_EMBED_DIM_IS_SEED = 4  # is_seed - domain CHI CO 2 gia tri {0,1},
                           # khong can chieu lon nhu cac field categorical
                           # khac (vocab lon hang tram gia tri)


# ================================================================
# CharVocab — GIONG HET convention l4_charcnn.c cu (build tu ky tu
# THAT xuat hien trong data train, KHONG dung ASCII truc tiep - xem
# comment bug #2 trong l4_charcnn.h ban cu, day la ly do vi sao KHONG
# duoc doi cach lam nay).
# ================================================================
class CharVocab:
    def __init__(self, texts, max_len=CHAR_MAX_LEN):
        chars = sorted(set("".join(t for t in texts if t)))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad/unknown
        self.max_len = max_len

    def encode(self, s):
        if not s:
            return [0] * self.max_len
        ids = [self.stoi.get(c, 0) for c in s[: self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def dump(self, path):
        with open(path, "w") as f:
            f.write(f"max_len={self.max_len}\n")
            for c, i in self.stoi.items():
                f.write(f"{i}\t{repr(c)}\n")


# ================================================================
# CatVocab — categorical id -> index lien tuc [1..N], 0 = unknown/pad.
# XAY DONG TU CHINH TAP TRAIN (khong cap cung 128/vv - ly do da noi
# voi architect: data nho, cap cung ra phan lon dong embedding chet).
# ================================================================
class CatVocab:
    def __init__(self, values):
        uniq = sorted(set(int(v) for v in values))
        self.stoi = {v: i + 1 for i, v in enumerate(uniq)}  # 0 = unknown

    def encode(self, v):
        return self.stoi.get(int(v), 0)

    def size(self):
        return len(self.stoi) + 1  # +1 cho slot 0 (unknown/pad)

    def dump(self, path):
        with open(path, "w") as f:
            for v, i in self.stoi.items():
                f.write(f"{i}\t{v}\n")


# ================================================================
# NumericNorm — log1p(|x|)*sign(x) roi standardize (mean/std tren
# TRAIN SET, CO DINH sau khi chot - infer C dung LAI DUNG so nay, xem
# canh bao dau file).
# ================================================================
class NumericNorm:
    def __init__(self, arr):
        # arr: list of [N_NUMERIC+1] (numeric[15] + ts_delta)
        import numpy as np
        a = np.array(arr, dtype=np.float64)
        signed_log = np.sign(a) * np.log1p(np.abs(a))
        self.mean = signed_log.mean(axis=0)
        self.std = signed_log.std(axis=0)
        self.std[self.std < 1e-6] = 1.0  # tranh chia 0 cho cot hang so

    def transform(self, vec):
        import numpy as np
        a = np.array(vec, dtype=np.float64)
        signed_log = np.sign(a) * np.log1p(np.abs(a))
        return ((signed_log - self.mean) / self.std).astype("float32")

    def dump(self, path):
        with open(path, "w") as f:
            f.write("mean\t" + "\t".join(f"{x:.10g}" for x in self.mean) + "\n")
            f.write("std\t" + "\t".join(f"{x:.10g}" for x in self.std) + "\n")


# ================================================================
# Doc jsonl story export (field "event_sequence" tu story_reconstruct.c)
# ================================================================
def load_stories(path):
    stories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "event_sequence" not in obj or not obj["event_sequence"]:
                continue  # story khong doc duoc gi (seed_node_id khong
                          # tim thay trong processes.bin, hoac component
                          # rong) - bo qua, KHONG coi la loi crash toan bo
            stories.append(obj)
    return stories


# ================================================================
# SUA (kien truc THAT dung labels_event_*.csv tu 15.py, KHONG PHAI
# labels_v2.csv/build_labels_v2.py cu nua - 2 file nay CO TEN GIONG
# NHAU nhung schema HOAN TOAN KHAC, gay nham lan neu khong doc ky):
#
#   labels_v2.csv (CU, DA BO, build_labels_v2.py):
#     seed_node_id,label,event_start_idx,event_end_idx,total_events,
#     match_scenario,turn_point_method
#     -> 1 dong = 1 DOAN (segment) CAT LIEN TUC tu event_sequence toan
#     cuc (GLOBAL index range). Doan nay CO BUG RO NHAN QUA ACTOR KHAC
#     da ghi trong docstring 15.py: dai lien tuc [start:end] trong
#     event_sequence (da merge cross-process theo timestamp) CHAC CHAN
#     lan events cua actor KHAC chen giua, khong chi rieng actor dang
#     xet.
#
#   labels_event_theia_e3.csv (MOI, 15.py that su xuat ra):
#     seed_node_id,event_idx,actor_idx,node_id,subject_uuid,label,
#     match_scenario,method
#     -> 1 dong = 1 EVENT rieng le, DA GAN label THEO TUNG ACTOR (turn-
#     point rieng cho moi actor, KHONG PHAI global cutoff chung ca
#     story). event_idx la GLOBAL index trong event_sequence (dung
#     chung khong gian voi story['event_sequence']).
#
# FIX: build_segments() ben duoi KHONG con cat theo [start:end] LIEN
# TUC nua (se tai lap dung bug rò nhan qua actor khac) - thay vao do
# LOC (filter) CHI LAY cac event_idx THUOC DUNG actor_idx dang xet
# (dung parse_actor_idx_from_debug_line() giong het parse_actor_idx()
# trong 15.py, doc tu event_sequence_debug), roi tach thanh <=2 doan
# benign/malicious THEO CHINH actor do (khong phai theo toan bo story)
# - vua khop dung logic turn-point per-actor cua 15.py, vua tranh hoan
# toan viec lan event cua actor khac vao 1 sample.
# ================================================================
import re as _re

_ACTOR_PREFIX_RE = _re.compile(r"^\[P(\d+)\]")


def parse_actor_idx_from_debug_line(line):
    """Giong het parse_actor_idx() trong 15.py - doc prefix '[P<n>]' cua
    1 dong event_sequence_debug, tra ve actor_idx (int) hoac None neu
    dong khong dung dinh dang nay."""
    m = _ACTOR_PREFIX_RE.match(line)
    if not m:
        return None
    return int(m.group(1))


def build_actor_idx_map(story):
    """Tra ve list actor_idx_by_global_idx (song song voi event_sequence,
    cung do dai) tu event_sequence_debug cua story - vi tri i = actor_idx
    cua event thu i trong event_sequence, hoac None neu dong debug khong
    parse duoc (an toan bo qua, event nay se khong duoc gan cho actor
    nao ca khi loc theo actor). Story KHONG CO event_sequence_debug (VD
    build voi cau hinh tat debug de tiet kiem dung luong) -> tra ve list
    toan None, build_segments() se tu dong bo qua CANH BAO ro rang thay
    vi am tham cat sai."""
    debug_lines = story.get("event_sequence_debug") or []
    n = len(story.get("event_sequence") or [])
    out = [None] * n
    for i in range(min(n, len(debug_lines))):
        out[i] = parse_actor_idx_from_debug_line(debug_lines[i])
    return out


def load_labels(path):
    """Doc labels_event_*.csv (dang MOI, tu 15.py THAT SU xuat ra - xem
    giai thich schema o comment tren). Tra ve list dict, 1 dict/1 dong
    CSV (1 event), CHUA gop nhom - viec gop nhom theo (seed_node_id,
    actor_idx) thanh segment lam trong build_segments()."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        required_cols = {"seed_node_id", "event_idx", "actor_idx", "label"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"[LOI] labels.csv thieu cot {missing} - file nay co dung la "
                f"output that su cua 15.py (build_labels_event_level.py) khong? "
                f"Header hien tai: {reader.fieldnames}. Neu day la labels_v2.csv "
                f"CU (build_labels_v2.py, cot event_start_idx/event_end_idx) - "
                f"file do DA BI THAY THE, dung dung labels_event_*.csv moi.")
        for row in reader:
            rows.append({
                "seed_node_id": int(row["seed_node_id"]),
                "event_idx": int(row["event_idx"]),
                "actor_idx": int(row["actor_idx"]),
                "label": int(row["label"]),
            })
    return rows


def build_segments(stories, label_rows, max_seq_len):
    """SUA (lan 2, theo yeu cau architect: PER-EVENT THAT, KHONG CAT
    thanh 2 doan benign/malicious nua) - xem giai thich dai o load_
    labels()/comment tren ve schema labels.csv MOI. Logic:

      1. Nhom label_rows (1 dong/event) theo (seed_node_id, actor_idx).
      2. Voi moi nhom, LOC (filter) event_sequence CHI LAY dung cac
         event_idx thuoc actor_idx nay (dung build_actor_idx_map(), doi
         chieu voi event_idx trong labels.csv de XAC NHAN CHEO - neu
         actor_idx cua map va cua labels.csv LECH NHAU o 1 event_idx
         nao do, DAY LA BUG THAT - CANH BAO RO RANG, khong am tham dung
         sai du lieu).
      3. KHONG con tach thanh benign-segment/malicious-segment nua -
         GIU NGUYEN 1 sample DUY NHAT = TOAN BO chuoi event cua actor
         nay (da loc, theo dung thu tu thoi gian goc, cap boi
         max_seq_len), kem theo 1 MANG NHAN rieng cho tung event
         (per_event_labels[i] tuong ung events[i]) - lay THANG tu cot
         "label" cua labels.csv cho DUNG event_idx do, KHONG suy dien/
         gop gi ca.

    CANH BAO ts_delta SAU KHI LOC (da bao architect): ts_delta goc do C
    tinh la khoang cach voi event NGAY TRUOC no trong CHUOI MERGE TOAN
    CUC (co the la event cua actor KHAC), khong phai voi event truoc do
    CUA CHINH actor nay. Sau khi loc chi con event cua 1 actor, ts_delta
    giua 2 event lien tiep TRONG SAMPLE co the KHONG con dung nghia
    "khoang cach that giua 2 event cua actor nay" nua - CHI event dau
    tien (ep ve 0.0, quy uoc "khong co lich su truoc do trong sample")
    la chac chan dung, cac vi tri con lai CO THE mang gia tri stale.
    NEU event dict co field timestamp THO (ngoai ts_delta da tinh san),
    NEN recompute ts_delta = timestamp[i] - timestamp[i-1] SAU KHI loc,
    thay vi dung nguyen gia tri cu - hien tai CHUA lam buoc nay vi chua
    xac nhan duoc event dict co field timestamp tho hay khong, dung lai
    o muc "dung tam gia tri cu + canh bao" cho toi khi confirm.

    Tra ve: list of (events_list, labels_list) - 2 list CUNG DO DAI,
    labels_list[i] la nhan CUA DUNG event thu i (khong phai 1 label
    chung cho ca sample nua)."""
    story_by_seed = {}
    for st in stories:
        sid = st.get("seed_node_id")
        if sid is not None:
            story_by_seed[sid] = st

    # MOI (RAM): tinh actor_map 1 LAN cho MOI story (cache theo sid),
    # roi POP NGAY "event_sequence_debug" khoi story dict - field nay
    # (1 dong text/event, co the toi 2048 dong/story) CHI can dung THOANG
    # QUA de dung actor_map, KHONG can giu lai sau do. Neu KHONG pop, no
    # bi giu trong RAM SUOT qua trinh train (story_by_seed/stories van
    # song qua het main()) - day la nguyen nhan chinh gay ap luc RAM he
    # thong (KHONG PHAI GPU VRAM) tren Colab 12GB khi dataset lon. Cache
    # actor_map giup KHONG phai parse lai debug line nhieu lan neu 1
    # story co NHIEU actor_idx (nhieu group trong groups.items()).
    actor_map_by_sid = {}
    for sid, st in story_by_seed.items():
        actor_map_by_sid[sid] = build_actor_idx_map(st)
        st.pop("event_sequence_debug", None)

    # gom nhom (seed_node_id, actor_idx) -> list (event_idx, label)
    groups = {}
    for row in label_rows:
        key = (row["seed_node_id"], row["actor_idx"])
        groups.setdefault(key, []).append((row["event_idx"], row["label"]))

    out = []
    n_skipped_no_story = 0
    n_skipped_empty = 0
    n_mismatch_actor = 0
    n_truncated = 0
    n_cut_malicious = 0
    n_cut_benign = 0

    for (sid, actor_idx), items in groups.items():
        st = story_by_seed.get(sid)
        if st is None:
            n_skipped_no_story += 1
            continue
        full_seq = st.get("event_sequence") or []
        actor_map = actor_map_by_sid[sid]

        items.sort(key=lambda x: x[0])  # dam bao dung thu tu event_idx

        # ---- doi chieu cheo: event_idx trong labels.csv PHAI thuoc
        # dung actor_idx theo chinh story hien tai (an toan truoc truong
        # hop story da duoc build lai/khac ban voi luc sinh labels.csv) ----
        filtered_events = []
        filtered_labels = []
        for global_idx, lbl in items:
            if global_idx < 0 or global_idx >= len(full_seq):
                n_mismatch_actor += 1
                continue
            actual_actor = actor_map[global_idx] if global_idx < len(actor_map) else None
            if actual_actor != actor_idx:
                n_mismatch_actor += 1
                continue
            filtered_events.append(dict(full_seq[global_idx]))
            filtered_labels.append(lbl)

        if not filtered_events:
            n_skipped_empty += 1
            continue

        if len(filtered_events) > max_seq_len:
            n_truncated += 1
            # MOI: dem SO NHAN MALICIOUS (label=1) THAT SU bi mat trong
            # phan bi cat (duoi cung, tu vi tri max_seq_len tro di) -
            # "measure before fix": thay vi doan impact, dem THAT xem
            # truncation co cat mat ground-truth malicious nao khong.
            cut_labels = filtered_labels[max_seq_len:]
            n_cut_malicious += sum(1 for l in cut_labels if l == 1)
            n_cut_benign += sum(1 for l in cut_labels if l == 0)
        filtered_events = filtered_events[:max_seq_len]
        filtered_labels = filtered_labels[:max_seq_len]

        filtered_events[0]["ts_delta"] = 0.0   # xem canh bao ts_delta tren
        out.append((filtered_events, filtered_labels))

    if n_skipped_no_story:
        print(f"[WARN] {n_skipped_no_story} nhom (seed,actor) trong labels.csv khong tim "
              f"thay seed_node_id tuong ung trong stories.jsonl - bo qua (kiem tra lai 2 "
              f"file co dung cap khong)", file=sys.stderr)
    if n_skipped_empty:
        print(f"[WARN] {n_skipped_empty} nhom (seed,actor) loc ra RONG sau khi doi chieu "
              f"actor_idx - bo qua", file=sys.stderr)
    if n_mismatch_actor:
        print(f"[CANH BAO] {n_mismatch_actor} dong labels.csv co actor_idx KHONG KHOP "
              f"voi event_sequence_debug cua story hien tai (hoac event_idx vuot qua do "
              f"dai event_sequence) - da BO QUA cac dong nay, KHONG dung de tranh lan "
              f"event sai actor vao sample. Neu so nay LON, kiem tra lai stories.jsonl va "
              f"labels.csv co phai sinh tu CUNG 1 lan chay avro_main.c khong (story build "
              f"lai se doi thu tu/index, lam labels.csv cu (sinh tu ban truoc) khong con "
              f"khop).", file=sys.stderr)
    if n_truncated:
        print(f"[WARN] {n_truncated} sample bi cat bot vi dai hon max_seq_len={max_seq_len} "
              f"- CAC EVENT BI CAT (o cuoi) MAT LUON NHAN CUA CHUNG, khong anh huong nhan "
              f"cac event con lai trong sample.", file=sys.stderr)
        if n_cut_malicious > 0:
            print(f"[CANH BAO NGHIEM TRONG] Trong so do, {n_cut_malicious} EVENT "
                  f"MALICIOUS (ground-truth label=1) THAT SU BI MAT vi nam sau vi tri "
                  f"max_seq_len={max_seq_len} - day la MAT DU LIEU THAT, anh huong "
                  f"recall neu dung tap nay danh gia. Can tang max_seq_len (kem sua ca "
                  f"L4_SEQ_LIVE_MAX_EVENTS ben C + recompile, xem giai thich trong "
                  f"l4_feature_extract.h) hoac doi cach cat (VD giu N event MOI NHAT "
                  f"thay vi N event DAU) - QUYET DINH nay can architect chot.",
                  file=sys.stderr)
        else:
            print(f"[INFO] Trong {n_truncated} sample bi cat, phan bi cat CHI toan "
                  f"benign ({n_cut_benign} event benign, 0 event malicious) - truncation "
                  f"KHONG lam mat ground-truth nao ca, an toan ve mat recall voi "
                  f"max_seq_len={max_seq_len} hien tai.", file=sys.stderr)
    return out


# ================================================================
# MOI (STREAMING) - thay the load_stories()+load_labels()+build_segments()
# o TREN khi dataset qua lon de load HET vao RAM 1 luc (VD TRACE E3 ~9GB,
# THEIA E5 ~41GB raw JSONL - uoc luong peak RAM ~2.7x file .jsonl neu
# dung load_stories() cu, tinh tu so lieu THAT: THEIA E3 2.5GB -> peak
# RSS 7GB - ap dung ty le nay: TRACE E3 9GB -> ~25GB, THEIA E5 41GB ->
# ~112GB, CA HAI DEU VUOT XA 12GB RAM Colab -> OOM CHAC CHAN, khong phai
# "co the").
#
# 3 ham CU (load_stories/load_labels/build_segments) VAN GIU NGUYEN o
# tren - KHONG xoa, van dung duoc cho debug/smoke-test tren may nhieu
# RAM hoac dataset nho (docstring dau file: "~20 story luc dau"), NHUNG
# main() TU GIO KHONG GOI 3 HAM DO NUA - GOI HAM MOI stream_build_
# segments() ben duoi.
#
# QUYET DINH KIEN TRUC (tu chon, theo dung nguyen tac "single
# comprehensive script" da thong nhat voi architect - KHONG lam 2-pass
# qua file .jsonl neu KHONG bat buoc):
#   - build_segments() CU dung story_by_seed[sid] = st (GHI DE neu 2
#     dong co CUNG seed_node_id - "dong CUOI CUNG thang").
#   - Streaming KHONG THE biet "dong nao la cuoi cung" ma KHONG doc het
#     file TRUOC (2-pass, ton 2x thoi gian I/O/CPU tren file 9-41GB).
#   - THAY VI 2-pass, dua vao BAT BIEN KIEN TRUC DA CHOT cua CHINH du
#     an nay: "node_id la khoa dinh danh CHINH, tang don dieu, 1-1 voi
#     subject_uuid, KHONG BAO GIO tai su dung" (xem memory #Identity).
#     seed_node_id LA 1 gia tri node_id -> VE LY THUYET KHONG THE co 2
#     dong trung seed_node_id trong 1 file .jsonl xuat tu 1 lan chay
#     avro_main.c hop le.
#   - Streaming 1-PASS: xu ly dong CO seed_node_id can dung DAU TIEN gap
#     duoc, CAC dong SAU co CUNG sid (neu co - KHONG LE THEO BAT BIEN
#     TREN) se bi BO QUA + IN CANH BAO NGHIEM TRONG (KHONG am tham nhan,
#     KHONG am tham bo qua ma khong noi gi - day la dau hieu file loi
#     hoac bi ghep tu nhieu lan chay avro_main.c khac nhau, CAN BAO
#     ARCHITECT NGAY neu thay canh bao nay xuat hien).
# ================================================================
def _iter_jsonl_stories(path):
    """Generator - yield (line_index, story_dict) TUNG DONG MOT, giong
    het bo loc cua load_stories() cu (bo qua dong rong/khong co
    event_sequence). KHONG BAO GIO giu qua 1 story trong RAM cung luc -
    story cu bi xa (GC thu hoi) ngay khi vong lap goi ham nay chuyen
    sang dong tiep theo, vi khong co bien nao ben ngoai giu tham chieu
    lai no."""
    with open(path) as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "event_sequence" not in obj or not obj["event_sequence"]:
                continue
            yield line_index, obj


def load_labels_streaming(path):
    """THAY THE load_labels() cu - doc labels.csv TRUC TIEP thanh
    `groups` dict, KHONG qua buoc trung gian list cac dict/dong (list
    do voi labels.csv HANG TRIEU dong - VD 4.9M dong THEIA E3 THAT - tu
    no da chiem 1-2GB+ RAM khong can thiet, TRACE E3/THEIA E5 lon hon
    con nhieu hon nua). csv.DictReader von da doc theo dong (khong load
    het file vao RAM 1 luc) - CHI can khong tich luy them 1 list dict
    thua o tren no.

    Tra ve:
      groups: dict (seed_node_id, actor_idx) -> list (event_idx, label)
              DA SORT theo event_idx (giong het build_segments() cu).
      needed_sids: set cac seed_node_id CAN tim trong stories.jsonl -
              dung de PASS qua file .jsonl BIET NGAY story nao can giu,
              story nao bo qua NGAY LAP TUC (khong parse sau vao no)."""
    groups = {}
    n_rows = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        required_cols = {"seed_node_id", "event_idx", "actor_idx", "label"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"[LOI] labels.csv thieu cot {missing} - file nay co dung la "
                f"output that su cua 15.py (build_labels_event_level.py) khong? "
                f"Header hien tai: {reader.fieldnames}. Neu day la labels_v2.csv "
                f"CU (build_labels_v2.py, cot event_start_idx/event_end_idx) - "
                f"file do DA BI THAY THE, dung dung labels_event_*.csv moi.")
        for row in reader:
            n_rows += 1
            key = (int(row["seed_node_id"]), int(row["actor_idx"]))
            groups.setdefault(key, []).append(
                (int(row["event_idx"]), int(row["label"])))
    for key in groups:
        groups[key].sort(key=lambda x: x[0])  # dam bao dung thu tu event_idx
    needed_sids = set(sid for sid, _actor_idx in groups.keys())
    return groups, needed_sids, n_rows


def stream_build_segments(stories_path, labels_path, max_seq_len, tag="TRAIN"):
    """Ham MOI thay the load_stories()+build_segments() cu khi dataset
    qua lon (xem giai thich kien truc dai o tren dau section nay). Cung
    tra ve DUNG DINH DANG (list of (events_list, labels_list)) va DUNG
    cac dong canh bao/thong ke nhu build_segments() cu (n_skipped_*,
    n_mismatch_actor, n_truncated, n_cut_malicious/benign) - CHI khac
    CACH DOC DU LIEU (streaming tung dong thay vi load het), KHONG doi
    LOGIC LOC/CAT/GAN NHAN nao ca."""
    from collections import defaultdict

    groups, needed_sids, n_label_rows = load_labels_streaming(labels_path)
    if not groups:
        print(f"[WARN] {tag}: {labels_path} rong hoac khong co dong nao - "
              f"bo qua {stories_path}", file=sys.stderr)
        return [], {"n_stories_in_file": 0, "n_label_rows": n_label_rows}

    actor_idxs_by_sid = defaultdict(list)
    for sid, actor_idx in groups.keys():
        actor_idxs_by_sid[sid].append(actor_idx)

    out = []
    n_stories_in_file = 0       # tong so story hop le (co event_sequence)
                                 # doc duoc trong file - khop dung y nghia
                                 # len(stories_i) cua load_stories() cu
    processed_sids = set()      # sid DA xu ly xong - dung de phat hien
                                 # + bo qua dong TRUNG LAP (xem giai thich
                                 # bat bien node_id o tren)
    n_duplicate_sid_lines = 0
    n_skipped_empty = 0
    n_mismatch_actor = 0
    n_truncated = 0
    n_cut_malicious = 0
    n_cut_benign = 0
    required_max_seq_len = [0, None]  # [gia tri toi thieu de KHONG mat
                                       # malicious nao, seed_node_id ung
                                       # voi gia tri do] - list mutable de
                                       # cap nhat tu trong vong lap long,
                                       # xem giai thich tai diem gan gia
                                       # tri o duoi.
    HEARTBEAT_EVERY = 2000  # in tien do + RAM moi 2000 story XU LY (khop
                             # sid can dung) - file 41GB co the mat VAI
                             # CHUC PHUT, can tin hieu con song, khong
                             # phai doi cham het moi biet dang chay hay
                             # da treo.

    for line_index, story in _iter_jsonl_stories(stories_path):
        n_stories_in_file += 1
        sid = story.get("seed_node_id")
        if sid is None or sid not in needed_sids:
            continue  # story nay KHONG co trong labels.csv - bo qua NGAY,
                       # story bi xa khoi RAM luc vong lap qua dong tiep
                       # theo (khong bien nao giu tham chieu lai)

        if sid in processed_sids:
            n_duplicate_sid_lines += 1
            print(f"[CANH BAO NGHIEM TRONG - {tag}] seed_node_id={sid} XUAT "
                  f"HIEN TRUNG LAP nhieu hon 1 dong trong {stories_path} - "
                  f"VI PHAM bat bien 'node_id khong bao gio tai su dung' da "
                  f"chot voi architect. DANG DUNG dong DAU TIEN gap duoc, "
                  f"BO QUA dong nay (line {line_index}). Day la DAU HIEU "
                  f"FILE LOI hoac bi GHEP TU NHIEU LAN CHAY avro_main.c "
                  f"khac nhau - BAO ARCHITECT NGAY, KHONG tu y bo qua canh "
                  f"bao nay.", file=sys.stderr)
            continue
        processed_sids.add(sid)

        full_seq = story.get("event_sequence") or []
        actor_map = build_actor_idx_map(story)
        # story.pop khong can thiet o day nua (khac ban RAM cu) - ca
        # story sap bi xa HOAN TOAN khi vong lap qua dong tiep theo, pop
        # rieng 1 field khong tiet kiem gi them.

        for actor_idx in actor_idxs_by_sid[sid]:
            items = groups[(sid, actor_idx)]
            filtered_events = []
            filtered_labels = []
            for global_idx, lbl in items:
                if global_idx < 0 or global_idx >= len(full_seq):
                    n_mismatch_actor += 1
                    continue
                actual_actor = actor_map[global_idx] if global_idx < len(actor_map) else None
                if actual_actor != actor_idx:
                    n_mismatch_actor += 1
                    continue
                ev_copy = dict(full_seq[global_idx])
                # MOI (RAM): nen "numeric" tu Python list (15 float OBJECT
                # rieng + list container, ~536 byte) sang array.array('f')
                # (buffer C lien tuc, KHONG co object rieng cho tung so,
                # ~140 byte) - tiet kiem ~396 byte/event, uoc luong ~11GB
                # tren 25-43M event cua THEIA E5. AN TOAN 100% downstream:
                # StorySeqDataset.__getitem__() dung `list(ev["numeric"])`
                # - list() chay dung tren CA list LAN array.array (deu la
                # iterable), KHONG can sua gi o __getitem__ ca.
                if "numeric" in ev_copy:
                    ev_copy["numeric"] = array.array("f", ev_copy["numeric"])
                filtered_events.append(ev_copy)
                filtered_labels.append(lbl)

            if not filtered_events:
                n_skipped_empty += 1
                continue

            if len(filtered_events) > max_seq_len:
                n_truncated += 1
                cut_labels = filtered_labels[max_seq_len:]
                n_cut_malicious += sum(1 for l in cut_labels if l == 1)
                n_cut_benign += sum(1 for l in cut_labels if l == 0)
                if 1 in cut_labels:
                    # MOI: tim VI TRI malicious CUOI CUNG trong TOAN BO
                    # chuoi CHUA cat (filtered_labels day du, TRUOC khi
                    # ap [:max_seq_len]) - can gia tri nay+1 lam
                    # max_seq_len TOI THIEU de KHONG mat malicious nao
                    # (giu ca benign phia sau van co the bi cat, KHONG
                    # sao - chi malicious moi la thu KHONG duoc mat).
                    last_malicious_idx = max(
                        i for i, l in enumerate(filtered_labels) if l == 1)
                    needed = last_malicious_idx + 1
                    if needed > required_max_seq_len[0]:
                        required_max_seq_len[0] = needed
                        required_max_seq_len[1] = sid
            filtered_events = filtered_events[:max_seq_len]
            filtered_labels = filtered_labels[:max_seq_len]

            filtered_events[0]["ts_delta"] = 0.0
            out.append((filtered_events, filtered_labels))
        # story/full_seq/actor_map bi XA o day - het scope vong lap
        # ngoai (for line_index, story in _iter_jsonl_stories), KHONG
        # con bien nao giu tham chieu lai, GC thu hoi truoc khi doc
        # dong .jsonl TIEP THEO.

        if len(processed_sids) % HEARTBEAT_EVERY == 0:
            log_ram(f"{tag} streaming: da xu ly {len(processed_sids)}/"
                    f"{len(needed_sids)} seed_node_id can dung "
                    f"({stories_path})")

    missing_sids = needed_sids - processed_sids
    n_skipped_no_story = sum(
        1 for (sid, _actor_idx) in groups if sid in missing_sids)
    if missing_sids:
        print(f"[WARN] {tag}: {len(missing_sids)} seed_node_id trong "
              f"labels.csv KHONG tim thay trong {stories_path} - "
              f"{n_skipped_no_story} nhom (seed,actor) bi bo qua (kiem tra "
              f"lai 2 file co dung cap khong)", file=sys.stderr)
    if n_skipped_empty:
        print(f"[WARN] {tag}: {n_skipped_empty} nhom (seed,actor) loc ra "
              f"RONG sau khi doi chieu actor_idx - bo qua", file=sys.stderr)
    if n_mismatch_actor:
        print(f"[CANH BAO] {tag}: {n_mismatch_actor} dong labels.csv co "
              f"actor_idx KHONG KHOP voi event_sequence_debug cua story "
              f"hien tai (hoac event_idx vuot qua do dai event_sequence) - "
              f"da BO QUA cac dong nay, KHONG dung de tranh lan event sai "
              f"actor vao sample. Neu so nay LON, kiem tra lai stories.jsonl "
              f"va labels.csv co phai sinh tu CUNG 1 lan chay avro_main.c "
              f"khong.", file=sys.stderr)
    if n_truncated:
        print(f"[WARN] {tag}: {n_truncated} sample bi cat bot vi dai hon "
              f"max_seq_len={max_seq_len} - CAC EVENT BI CAT (o cuoi) MAT "
              f"LUON NHAN CUA CHUNG, khong anh huong nhan cac event con "
              f"lai trong sample.", file=sys.stderr)
        if n_cut_malicious > 0:
            print(f"[CANH BAO NGHIEM TRONG] {tag}: Trong so do, "
                  f"{n_cut_malicious} EVENT MALICIOUS (ground-truth "
                  f"label=1) THAT SU BI MAT vi nam sau vi tri "
                  f"max_seq_len={max_seq_len} - day la MAT DU LIEU THAT, "
                  f"anh huong recall neu dung tap nay danh gia. QUYET DINH "
                  f"xu ly can architect chot (tang max_seq_len + recompile "
                  f"C, hoac doi cach cat).", file=sys.stderr)
            print(f"[GOI Y - {tag}] De KHONG MAT MALICIOUS NAO trong "
                  f"{stories_path} (van co the mat benign o duoi, KHONG "
                  f"sao), can max_seq_len >= {required_max_seq_len[0]} "
                  f"(hien tai dang dung {max_seq_len}) - gia tri nay tinh "
                  f"THAT tu vi tri malicious CUOI CUNG trong sample can "
                  f"nhieu cho nhat (seed_node_id={required_max_seq_len[1]}). "
                  f"Day la CAN DUOI CHINH XAC cho DATASET NAY - datasets "
                  f"khac (VD dataset thu 2 trong --stories) co the can gia "
                  f"tri KHAC, xem dong [GOI Y] rieng cua tung dataset roi "
                  f"LAY SO LON NHAT trong tat ca truoc khi quyet dinh "
                  f"max_seq_len chung + sua L4_SEQ_LIVE_MAX_EVENTS ben C.",
                  file=sys.stderr)
        else:
            print(f"[INFO] {tag}: Trong {n_truncated} sample bi cat, phan "
                  f"bi cat CHI toan benign ({n_cut_benign} event benign, 0 "
                  f"event malicious) - truncation KHONG lam mat ground-"
                  f"truth nao ca, an toan ve mat recall voi "
                  f"max_seq_len={max_seq_len} hien tai.", file=sys.stderr)
    if n_duplicate_sid_lines:
        print(f"[CANH BAO TOM TAT - {tag}] Tong cong {n_duplicate_sid_lines} "
              f"dong bi bo qua vi seed_node_id TRUNG LAP - xem chi tiet tung "
              f"dong o cac dong CANH BAO NGHIEM TRONG phia tren.",
              file=sys.stderr)

    stats = {
        "n_stories_in_file": n_stories_in_file,
        "n_label_rows": n_label_rows,
        "n_sids_needed": len(needed_sids),
        "n_sids_found": len(processed_sids),
    }
    return out, stats


# ================================================================
# Dataset
# ================================================================
class StorySeqDataset(Dataset):
    """SUA (per-event) - nhan segments DA CAT SAN (xem build_segments()),
    MOI sample gio la (events_list, labels_list) - labels_list CUNG DO
    DAI voi events_list, moi vi tri co nhan RIENG."""

    def __init__(self, segments, cat_vocabs, char_vocab, num_norm):
        self.samples = segments  # list of (events_list, labels_list) -
                                  # DA cat + DA ap max_seq_len trong
                                  # build_segments(), 2 list CUNG DO DAI
        self.cat_vocabs = cat_vocabs
        self.char_vocab = char_vocab
        self.num_norm = num_norm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        events, labels = self.samples[idx]
        n = len(events)
        assert len(labels) == n, (
            f"[BUG] events va labels lech do dai ({n} vs {len(labels)}) - "
            f"loi nay khong duoc xay ra neu build_segments() dung, kiem tra lai")

        cat_tag = torch.zeros(n, dtype=torch.long)
        cat_syscall = torch.zeros(n, dtype=torch.long)
        cat_errno = torch.zeros(n, dtype=torch.long)
        cat_behavior = torch.zeros(n, dtype=torch.long)
        cat_is_seed = torch.zeros(n, dtype=torch.long)  # domain co dinh
                                        # {0,1}, feed THANG gia tri that
                                        # lam index, KHONG qua CatVocab
        numeric = torch.zeros(n, N_NUMERIC + 1, dtype=torch.float32)
        char_primary = torch.zeros(n, self.char_vocab.max_len, dtype=torch.long)
        char_secondary = torch.zeros(n, self.char_vocab.max_len, dtype=torch.long)

        for i, ev in enumerate(events):
            cat_tag[i] = self.cat_vocabs["tag"].encode(ev["tag"])
            cat_syscall[i] = self.cat_vocabs["syscall_id"].encode(ev["syscall_id"])
            cat_errno[i] = self.cat_vocabs["errno"].encode(ev["errno"])
            cat_behavior[i] = self.cat_vocabs["behavior"].encode(ev["behavior"])
            cat_is_seed[i] = int(ev.get("is_seed", 0))
            raw_num = list(ev["numeric"]) + [ev.get("ts_delta", 0.0)]
            numeric[i] = torch.tensor(self.num_norm.transform(raw_num))
            char_primary[i] = torch.tensor(self.char_vocab.encode(ev.get("path_primary")))
            char_secondary[i] = torch.tensor(self.char_vocab.encode(ev.get("path_secondary")))

        return {
            "cat_tag": cat_tag, "cat_syscall": cat_syscall,
            "cat_errno": cat_errno, "cat_behavior": cat_behavior,
            "cat_is_seed": cat_is_seed,
            "numeric": numeric,
            "char_primary": char_primary, "char_secondary": char_secondary,
            "labels": torch.tensor(labels, dtype=torch.float32),  # [n] - MOI, 1 nhan/event
            "seq_len": n,
        }


def log_ram(tag):
    """MOI (RAM). In CA 2 SO: peak RSS (high-water mark, KHONG BAO GIO
    giam - ru_maxrss tinh tu luc process bat dau) VA current RSS (RAM
    SONG THAT SU ngay bay gio, doc tu /proc/self/status VmRSS - CHI co
    tren Linux, Colab/WSL2 deu la Linux nen an toan). SUA (theo dung lo
    ngai cua architect ve OOM): chi in peak RSS truoc day GAY HIEU LAM -
    peak KHONG giam du del/gc.collect() da chay, nhin peak KHONG biet
    duoc del co thuc su giai phong RAM hay khong. current RSS moi la so
    PHAN ANH RUI RO OOM THAT SU tai thoi diem goi ham (RAM con trong may
    = tong RAM - current RSS, xap xi, chua tinh cache OS)."""
    rss_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rss_now_mb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_now_mb = int(line.split()[1]) / 1024
                    break
    except FileNotFoundError:
        pass  # khong phai Linux (hiem, Colab/WSL2 deu la Linux) - bo qua
    if rss_now_mb is not None:
        print(f"[RAM] {tag}: current RSS = {rss_now_mb:.0f} MB (RAM SONG "
              f"THAT SU ngay bay gio) | peak RSS = {rss_peak_mb:.0f} MB "
              f"(cao nhat TU TRUOC TOI GIO, KHONG giam du da del/gc)")
        # MOI: canh bao SOM neu current RSS da vuot nguong nguy hiem -
        # 12GB la RAM tong cua Colab T4 runtime (KHONG PHAI 100% danh cho
        # process nay, con OS/torch/cuda giu 1 phan) - ep vach an toan o
        # 9500MB (~80%) de anh THAY CANH BAO va co the ngat train som
        # (Runtime > Interrupt execution) THAY VI doi Colab tu kill process
        # (mat het progress, khong co traceback ro rang, chi bao "session
        # crashed"). Nguong nay UOC LUONG, KHONG phai gioi han cung.
        if rss_now_mb > 9500:
            print(f"[CANH BAO OOM SAP TOI] current RSS = {rss_now_mb:.0f} MB "
                  f"da VUOT 9.5GB/12GB (~80%) - RUI RO OOM CAO trong buoc "
                  f"TIEP THEO (dataset ke tiep hoac vong train). CAN NHAC "
                  f"ngat thu cong (Runtime > Interrupt execution) NGAY BAY "
                  f"GIO neu con dataset chua load xong, thay vi doi Colab tu "
                  f"crash session (session crash mat toan bo tien do, "
                  f"KHONG the resume tu checkpoint vi chua train xong).",
                  file=sys.stderr)
    else:
        print(f"[RAM] {tag}: peak RSS = {rss_peak_mb:.0f} MB (khong doc "
              f"duoc current RSS - khong phai Linux)")


def move_batch_to_device(batch, device):
    """Chuyen 1 batch (dict cac tensor tra ve tu collate_pad) sang device
    (GPU/CPU). CHI di chuyen TUNG BATCH tai thoi diem dung (khong move ca
    dataset/segments len GPU truoc) - giu toan bo dataset trong CPU RAM
    nhu cu, chi 1 batch nam tren GPU VRAM tai 1 thoi diem, tranh OOM khi
    RAM he thong gioi han 12GB (Colab). non_blocking=True chi co tac dung
    that su neu DataLoader dung pin_memory=True (xem main())."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def collate_pad(batch):
    """Pad ve max_len TRONG batch (khong phai global max_seq_len) - tiet
    kiem tinh toan cho batch co story ngan. Padding = 0 het (tag=0/
    syscall=0/... deu la slot 'unknown/pad', numeric=0 sau standardize
    KHONG dung 0 that vi da chuan hoa - dung mask rieng, xem attention_
    mask ben duoi, TCN dung causal conv nen padding cuoi KHONG lam lo
    thong tin nguoc ve truoc, an toan de mask sau khi pool).

    SUA (per-event): "labels" gio la [B, max_len] (1 gia tri/event),
    KHONG PHAI [B] (1 gia tri/sample) nua - vi tri padding co labels=0
    NHUNG se bi LOAI KHOI LOSS boi "mask" (xem train loop, loss tinh
    CHI tren vi tri mask=True, khong phai 0 padding lam sai lech gradient)."""
    max_len = max(b["seq_len"] for b in batch)
    B = len(batch)

    def pad_1d(key, dtype):
        out = torch.zeros(B, max_len, dtype=dtype)
        for i, b in enumerate(batch):
            n = b["seq_len"]
            out[i, :n] = b[key]
        return out

    def pad_2d(key, dim2, dtype):
        out = torch.zeros(B, max_len, dim2, dtype=dtype)
        for i, b in enumerate(batch):
            n = b["seq_len"]
            out[i, :n] = b[key]
        return out

    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask[i, : b["seq_len"]] = True

    return {
        "cat_tag": pad_1d("cat_tag", torch.long),
        "cat_syscall": pad_1d("cat_syscall", torch.long),
        "cat_errno": pad_1d("cat_errno", torch.long),
        "cat_behavior": pad_1d("cat_behavior", torch.long),
        "cat_is_seed": pad_1d("cat_is_seed", torch.long),  # MOI - pad=0
                                        # trung voi gia tri that "khong
                                        # phai seed", KHONG sao vi mask
                                        # rieng da loai vi tri padding
                                        # khoi pooling (xem docstring tren),
                                        # TCN causal cung khong nhin tuong
                                        # lai nen khong lo thong tin
        "numeric": pad_2d("numeric", N_NUMERIC + 1, torch.float32),
        "char_primary": pad_2d("char_primary", CHAR_MAX_LEN, torch.long),
        "char_secondary": pad_2d("char_secondary", CHAR_MAX_LEN, torch.long),
        "labels": pad_1d("labels", torch.float32),  # MOI - [B, max_len], 1 nhan/event
        "mask": mask,
    }


# ================================================================
# Model
# ================================================================
class CharCNNEncoder(nn.Module):
    """CharCNN — encode 1 string (path/argv) -> vector CHAR_EMBED_DIM.
    Ap dung DOC LAP cho tung event (khong biet gi ve thu tu chuoi -
    dung TCN o tang tren de lam viec do).

    KHOI PHUC (2026-08-11): BO HET phan chia chunk + torch.utils.
    checkpoint da thu truoc do - PHIEN BAN NAY CHUA TUNG duoc test
    tren GPU that (khong co GPU trong moi truong viet code), gay OOM
    LAP LAI NHIEU LAN khi dung that, LAM MAT THOI GIAN quy gia cua
    architect giua luc gap deadline. Quay ve DUNG BAN GOC DON GIAN -
    ban nay DA CHAY THAT THANH CONG tren Colab (epoch1=1.9378,
    epoch2=0.2980, epoch3=0.2366, KHONG OOM, dung --batch_size mac
    dinh=4). Neu can toi uu VRAM/toc do sau nay, PHAI test tren GPU
    THAT truoc khi giao lai, khong doan mo nua."""

    def __init__(self, vocab_size, embed_dim=CHAR_EMBED_DIM):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size + 1, 16, padding_idx=0)
        self.conv1 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [..., char_max_len] long -> flatten batch*seq de chay CNN
        # 1 lan cho ca batch (re hon loop tung event)
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        e = self.char_embed(x).transpose(1, 2)  # [N, 16, L]
        h = F.relu(self.conv1(e))
        h = F.relu(self.conv2(h))
        h = h.max(dim=2).values  # global max pool tren truc ky tu
        return h.reshape(*orig_shape, -1)


class CausalTCNBlock(nn.Module):
    """1 lop dilated causal conv1d - CHI nhin ve QUA KHU (padding lech
    trai + cat bo phan du o phai), dung tinh chat 'chuoi co thu tu' ma
    architect yeu cau, khac LSTM ve co che nhung GIU DUNG rang buoc
    nhan-qua thoi gian."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: [B, C, T]
        h = self.conv(x)
        h = h[:, :, : -self.pad] if self.pad > 0 else h  # cat phan du o
                                                            # PHAI (tuong lai)
                                                            # - dam bao causal
        h = F.relu(self.norm(h))
        return x + h  # residual


class L4SeqClassifier(nn.Module):
    def __init__(self, cat_sizes, char_vocab_size):
        super().__init__()
        self.embed_tag = nn.Embedding(cat_sizes["tag"], CAT_EMBED_DIM_SMALL, padding_idx=0)
        self.embed_syscall = nn.Embedding(cat_sizes["syscall_id"], CAT_EMBED_DIM_BIG, padding_idx=0)
        self.embed_errno = nn.Embedding(cat_sizes["errno"], CAT_EMBED_DIM_SMALL, padding_idx=0)
        self.embed_behavior = nn.Embedding(cat_sizes["behavior"], CAT_EMBED_DIM_SMALL, padding_idx=0)
        self.embed_is_seed = nn.Embedding(2, CAT_EMBED_DIM_IS_SEED)  # MOI -
                                        # domain co dinh {0,1}, KHONG dung
                                        # padding_idx=0 (0 la gia tri THAT
                                        # "khong phai seed", khong phai
                                        # pad/unknown nhu 4 field kia -
                                        # xem giai thich cat_is_seed dau
                                        # file, vi sao field nay khac ban
                                        # chat voi 4 categorical con lai)

        self.charcnn = CharCNNEncoder(char_vocab_size)

        # event_dim TINH DONG tu chinh cac embedding/numeric da khoi tao
        # o tren (KHONG hard-code tong bang literal - da tung bi bat loi
        # "153" ghi cung trong comment cu trong khi thuc chat la cong
        # thuc, tranh lap lai kieu bug "so lieu bia chua do" da xay ra
        # 1 lan voi benchmark 65x).
        event_dim = (
            CAT_EMBED_DIM_SMALL     # tag
            + CAT_EMBED_DIM_BIG     # syscall_id
            + CAT_EMBED_DIM_SMALL   # errno
            + CAT_EMBED_DIM_SMALL   # behavior
            + CAT_EMBED_DIM_IS_SEED # is_seed (MOI)
            + (N_NUMERIC + 1)       # numeric[15] + ts_delta
            + CHAR_EMBED_DIM * 2    # path_primary + path_secondary
        )

        self.input_proj = nn.Linear(event_dim, 64)

        self.tcn_layers = nn.ModuleList([
            CausalTCNBlock(64, kernel_size=3, dilation=d) for d in [1, 2, 4, 8]
        ])

        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )  # SUA (per-event): input 64 (khong con *2 tu mean+max pool
           # nua) - head gio ap dung TRUC TIEP tren tung vi tri thoi
           # gian h[:,t,:], KHONG qua pooling toan chuoi. Vi TCN dung
           # causal conv (KHONG nhin tuong lai), h[:,t,:] van CHI phu
           # thuoc event 0..t - dung tinh chat "du doan tai thoi diem t
           # CHI dua vao qua khu" ma per-event classification can.

    def forward(self, batch):
        cat = torch.cat([
            self.embed_tag(batch["cat_tag"]),
            self.embed_syscall(batch["cat_syscall"]),
            self.embed_errno(batch["cat_errno"]),
            self.embed_behavior(batch["cat_behavior"]),
            self.embed_is_seed(batch["cat_is_seed"]),
        ], dim=-1)  # [B, T, 16+32+16+16+4]

        str_primary = self.charcnn(batch["char_primary"])      # [B, T, 32]
        str_secondary = self.charcnn(batch["char_secondary"])  # [B, T, 32]

        event_vec = torch.cat([cat, batch["numeric"], str_primary, str_secondary], dim=-1)
        h = self.input_proj(event_vec)  # [B, T, 64]

        h = h.transpose(1, 2)  # [B, 64, T] cho Conv1d
        for layer in self.tcn_layers:
            h = layer(h)
        h = h.transpose(1, 2)  # [B, T, 64]

        # SUA (per-event): BO HAN mean/max pool toan chuoi - ap head
        # TRUC TIEP tren tung vi tri T, tra ve logit [B, T] (1 gia tri/
        # event) thay vi [B] (1 gia tri/sample). Padding position van
        # duoc tinh logit binh thuong (khong sao, se bi loai khoi loss
        # boi mask o ngoai train loop, khong anh huong gradient).
        logit = self.head(h).squeeze(-1)  # [B, T]
        return logit


# ================================================================
# Train loop
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", action="append", required=True,
                     help="Story JSONL DUNG DE TRAIN - CO THE lap lai flag nay "
                          "NHIEU LAN de gop nhieu dataset vao 1 tap train (VD "
                          "LOSO: --stories theia.jsonl --stories trace.jsonl de "
                          "train tren CA HAI, roi --eval_stories theia_e5.jsonl "
                          "de test tren dataset con lai). THU TU PHAI khop 1-1 "
                          "voi --labels (file --stories thu i ghep voi --labels "
                          "thu i).")
    ap.add_argument("--labels", action="append", required=True,
                     help="Labels CSV (tu 15.py) DUNG DE TRAIN - lap lai flag nay "
                          "THEO DUNG SO LUONG VA THU TU voi --stories.")
    ap.add_argument("--eval_stories", action="append", default=[],
                     help="MOI (LOSO) - Story JSONL DUNG DE DANH GIA (held-out), "
                          "KHONG dua vao train. CO THE lap lai flag nay nhieu lan "
                          "(it dung hon --stories, thuong chi 1 dataset con lai). "
                          "De trong (mac dinh) = KHONG danh gia held-out gi ca, "
                          "chi in loss tren tap TRAIN (khong du de bao cao trong "
                          "paper, xem canh bao cuoi script). Dataset KHAC hoan "
                          "toan voi tat ca --stories (dung LOSO that) = cross-host "
                          "- xem canh bao rieng ve TRACE/THEIA-E5 cmdLine null "
                          "trong phan output eval.")
    ap.add_argument("--eval_labels", action="append", default=[],
                     help="Labels CSV cho --eval_stories - so luong va thu tu "
                          "PHAI khop 1-1 voi --eval_stories.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--hnm_epochs", type=int, default=0,
                     help="MOI (Hard Negative Mining). So epoch train THEM sau "
                          "khi vong train BINH THUONG (--epochs) da xong, tap "
                          "trung vao cac sample BENIGN 'kho' (model tu cham lai "
                          "tap TRAIN, benign nao bi predict xac suat CAO nhat = "
                          "de nham voi malicious nhat). Mac dinh 0 = TAT (khong "
                          "chay them gi ca, giu nguyen hanh vi cu). Day la OHEM "
                          "(Online Hard Example Mining) 'in-sample' - mine TU "
                          "CHINH tap train hien co, KHONG sinh them du lieu benign "
                          "moi tu C pipeline (ban 'day du hon' can tang "
                          "EDR_STORY_AUTOSEED_TOPK_C ben avro_main.c va chay lai "
                          "15.py de co pool benign lon hon truoc khi mine - viec "
                          "lon hon, CAN ARCHITECT QUYET DINH neu muon lam, KHONG "
                          "tu y lam trong pham vi sua file nay).")
    ap.add_argument("--hnm_topk_frac", type=float, default=0.3,
                     help="Ty le sample BENIGN 'kho nhat' duoc oversample trong "
                          "cac epoch HNM (VD 0.3 = 30%% sample benign co diem "
                          "predict cao nhat). Chi co tac dung neu --hnm_epochs>0.")
    ap.add_argument("--hnm_oversample_weight", type=float, default=5.0,
                     help="He so tang trong so lay mau (WeightedRandomSampler) "
                          "cho cac sample 'hard negative' so voi sample thuong "
                          "(weight=1.0) trong cac epoch HNM. Chi co tac dung "
                          "neu --hnm_epochs>0.")
    ap.add_argument("--device", type=str, default="auto",
                     choices=["auto", "cuda", "cpu"],
                     help="MOI. 'auto' (mac dinh) = dung GPU neu torch.cuda."
                          "is_available(), khong thi fallback CPU tu dong + in "
                          "canh bao. 'cuda' = BAT BUOC GPU, bao loi va thoat neu "
                          "khong co (tranh am tham chay CPU cham ma khong biet). "
                          "'cpu' = ep CPU du GPU co san (debug/so sanh). Dataset/"
                          "segments VAN nam trong CPU RAM nhu cu bat ke gia tri "
                          "nay - CHI tung batch duoc chuyen len GPU VRAM tai thoi "
                          "diem train/eval (xem move_batch_to_device()), khong "
                          "doi cach load data, tranh OOM voi gioi han RAM 12GB.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[LOI] --device cuda nhung torch.cuda.is_available()=False - "
              "khong co GPU kha dung (kiem tra lai runtime Colab: Runtime > "
              "Change runtime type > GPU, hoac driver/CUDA build cua torch)",
              file=sys.stderr)
        sys.exit(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[WARN] Khong tim thay GPU (torch.cuda.is_available()=False) - "
                  "fallback CPU, train se CHAM hon dang ke. Neu dang chay Colab, "
                  "kiem tra Runtime > Change runtime type > GPU truoc khi chay lai.")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] Dung GPU: {gpu_name} (~{vram_gb:.1f}GB VRAM)")

    if len(args.stories) != len(args.labels):
        print(f"[LOI] so luong --stories ({len(args.stories)}) khac so luong "
              f"--labels ({len(args.labels)}) - PHAI khop 1-1 theo thu tu",
              file=sys.stderr)
        sys.exit(1)
    if len(args.eval_stories) != len(args.eval_labels):
        print(f"[LOI] so luong --eval_stories ({len(args.eval_stories)}) khac "
              f"so luong --eval_labels ({len(args.eval_labels)}) - PHAI khop 1-1",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # MOI: in TOM TAT fold ngay dau tien (TRUOC khi load bat cu gi) - de
    # thay NGAY dang chay fold nao (VD "train theia_e3+trace_e3 -> eval
    # theia_e5" cho LOSO Fold 3) ma KHONG phai doc rai rac tung dong
    # [INFO] TRAIN/EVAL o duoi. Dung os.path.basename() cho de doc (full
    # path qua dai), nhung van in FULL PATH ben duoi 1 lan de doi chieu
    # dung file neu can debug.
    # MOI: in TOM TAT fold ngay dau tien (TRUOC khi load bat cu gi) - de
    # thay NGAY dang chay fold nao (VD "train theia_e3+trace_e3 -> eval
    # theia_e5" cho LOSO Fold 3) ma KHONG phai doc rai rac tung dong
    # [INFO] TRAIN/EVAL o duoi.
    #
    # SUA (phat hien qua test thu that su - "measure before fix"): CHI
    # dung os.path.basename() la SAI voi cau truc thu muc du an nay - cac
    # file story_export.jsonl cua THEIA E3/TRACE E3/THEIA E5 RAT CO THE
    # dung CHUNG 1 TEN FILE (basename giong het nhau), chi khac THU MUC
    # CHA (VD .../theia_trace_e3/story_export.jsonl vs .../theia_trace_e5/
    # story_export.jsonl) - basename don le se in ra "story_export.jsonl,
    # story_export.jsonl" KHONG PHAN BIET DUOC, phan tac dung cua chinh
    # tinh nang nay. Dung 2 CAP path cuoi (ten thu muc cha + ten file).
    def _short_path(p):
        parent = os.path.basename(os.path.dirname(p.rstrip("/")))
        leaf = os.path.basename(p)
        return f"{parent}/{leaf}" if parent else leaf

    train_names = [_short_path(p) for p in args.stories]
    eval_names = [_short_path(p) for p in args.eval_stories] if args.eval_stories else []
    print("=" * 70)
    print(f"[FOLD] TRAIN ({len(train_names)} dataset): {', '.join(train_names)}")
    if eval_names:
        overlap = set(args.stories) & set(args.eval_stories)
        fold_kind = "WITHIN-HOST/CO TRUNG FILE VOI TRAIN" if overlap else "CROSS-HOST (LOSO that)"
        print(f"[FOLD] EVAL/TEST ({len(eval_names)} dataset, {fold_kind}): "
              f"{', '.join(eval_names)}")
    else:
        print("[FOLD] EVAL/TEST: KHONG CO (--eval_stories rong) - chi in "
              "train_loss, KHONG co so lieu held-out nao ca.")
    print(f"[FOLD] --stories full path: {args.stories}")
    if args.eval_stories:
        print(f"[FOLD] --eval_stories full path: {args.eval_stories}")
    print("=" * 70)

    # ---- MOI - GOM NHIEU dataset train (moi cap stories/labels DUOC CAT
    # RIENG boi build_segments() cua CHINH NO, roi moi noi ket qua lai -
    # AN TOAN voi trung seed_node_id giua cac dataset khac nhau (VD THEIA
    # va TRACE CO THE co seed_node_id trung nhau ve mat SO HOC, vi day la
    # 2 khong gian id DOC LAP tu 2 lan chay avro_main.c khac nhau) - vi
    # story_by_seed dict duoc build_segments() tao MOI LAN GOI RIENG (chi
    # tu 1 cap stories/labels), KHONG co nguy co 1 seed_node_id cua THEIA
    # vo tinh khop nham story cua TRACE khi gop chung. ----
    segments = []
    for stories_path, labels_path in zip(args.stories, args.labels):
        log_ram(f"TRUOC khi xu ly {os.path.basename(stories_path)}")
        # SUA (STREAMING - xem giai thich kien truc day du o dinh nghia
        # stream_build_segments()): KHONG con load_stories()+load_labels()
        # +build_segments() rieng le (doc HET file vao RAM truoc) nua -
        # goi 1 ham DUY NHAT xu ly TUNG DONG .jsonl, KHONG BAO GIO giu
        # qua 1 story trong RAM cung luc. Bat buoc voi dataset lon (VD
        # TRACE E3 ~9GB, THEIA E5 ~41GB raw) - cach cu (load het) uoc
        # luong OOM CHAC CHAN tren Colab 12GB RAM (xem thao luan RAM da
        # thong nhat voi architect).
        segments_i, stats_i = stream_build_segments(
            stories_path, labels_path, args.max_seq_len, tag="TRAIN")
        print(f"[INFO] TRAIN: doc {stats_i['n_stories_in_file']} story, "
              f"{stats_i['n_label_rows']} dong labels.csv tu {stories_path} "
              f"/ {labels_path}")
        print(f"[INFO] TRAIN: {stories_path} -> {len(segments_i)} sample sau cat "
              f"({stats_i['n_sids_found']}/{stats_i['n_sids_needed']} "
              f"seed_node_id can dung da tim thay)")
        segments.extend(segments_i)
        del segments_i  # segments DA gop vao list ngoai (`segments`) qua
                         # extend() - xoa bien tam ngay, KHONG doi vong
                         # lap sau ghi de (dataset TIEP THEO co the lon
                         # hon nhieu, giai phong SOM van tot hon).
        log_ram(f"sau khi xu ly xong {os.path.basename(stories_path)}")

    if not segments:
        print("[ERROR] khong cat duoc segment nao tu BAT KY cap --stories/--labels "
              "nao - kiem tra lai seed_node_id co khop giua stories.jsonl va "
              "labels.csv khong", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] TONG CONG TRAIN (gop {len(args.stories)} dataset): "
          f"{len(segments)} sample")

    all_events = [ev for events, _ in segments for ev in events]
    if not all_events:
        print("[ERROR] khong co event nao trong cac segment - kiem tra "
              "lai labels.csv co khop seed_node_id voi stories khong", file=sys.stderr)
        sys.exit(1)

    # QUAN TRONG (LOSO/tranh leakage): vocab (cat/char) va numeric norm
    # CHI duoc fit tren TAP TRAIN (DA GOP, neu nhieu dataset) - KHONG BAO
    # GIO fit lai/mo rong tren tap eval, du eval co gap category/ky tu la
    # (VD syscall chi xuat hien o dataset con lai, khong co trong tap
    # train) - dung QUY UOC "unknown" (index 0) cho cac gia tri eval
    # khong co trong vocab train, GIONG HET cach
    # model se xu ly khi deploy that (khong bao gio "thay truoc" du lieu
    # host moi).
    cat_vocabs = {
        "tag": CatVocab([ev["tag"] for ev in all_events]),
        "syscall_id": CatVocab([ev["syscall_id"] for ev in all_events]),
        "errno": CatVocab([ev["errno"] for ev in all_events]),
        "behavior": CatVocab([ev["behavior"] for ev in all_events]),
    }
    cat_sizes = {k: v.size() for k, v in cat_vocabs.items()}
    print(f"[INFO] cat vocab sizes (fit tren TRAIN): {cat_sizes}")

    all_strings = []
    for ev in all_events:
        if ev.get("path_primary"):
            all_strings.append(ev["path_primary"])
        if ev.get("path_secondary"):
            all_strings.append(ev["path_secondary"])
    char_vocab = CharVocab(all_strings)
    print(f"[INFO] char vocab size (fit tren TRAIN): {len(char_vocab.stoi)}")

    numeric_arr = [list(ev["numeric"]) + [ev.get("ts_delta", 0.0)] for ev in all_events]
    num_norm = NumericNorm(numeric_arr)
    del numeric_arr, all_strings, all_events  # MOI (RAM) - all_events la
                                # list THAM CHIEU toi event dict cua TAT
                                # CA segments, chi can cho vocab/norm fit
                                # (da xong o day) - xoa THAM CHIEU nay
                                # (KHONG xoa segments/dataset that su,
                                # events van song qua `segments` ma
                                # StorySeqDataset can), giam 1 ban sao
                                # danh sach du thua truoc khi vao vong
                                # train (dai nhat, chay lau nhat).
    log_ram("sau khi build xong vocab/norm (truoc khi vao vong train)")

    # MOI (KHAN CAP - Colab sap het gio): LUU NGAY vocab/norm ra dia luc
    # nay, KHONG doi den cuoi script (dong 1550 cu) - neu Colab ngat
    # giua chung (het gio session, disconnect), vocab/norm van la thu
    # BAT BUOC phai co de dung BAT KY checkpoint model nao ve sau (khong
    # co vocab = khong doc duoc model, du model.pt con nguyen). Cac file
    # nay KHONG doi trong suot qua trinh train (fit 1 lan duy nhat o
    # tren), nen luu som HOAN TOAN an toan, khong can luu lai lan nua o
    # cuoi (dong duplicate cu se ghi de bang gia tri GIONG HET, vo hai
    # nhung du thua - co the xoa sau, khong gap dung luc nay).
    os.makedirs(args.out_dir, exist_ok=True)
    for name, vocab in cat_vocabs.items():
        vocab.dump(os.path.join(args.out_dir, f"l4_seq_cat_{name}.vocab"))
    char_vocab.dump(os.path.join(args.out_dir, "l4_seq_char.vocab"))
    num_norm.dump(os.path.join(args.out_dir, "l4_seq_numeric_norm.txt"))
    print(f"[INFO] DA LUU vocab/norm ra {args.out_dir} NGAY BAY GIO (an toan "
          f"du session co ngat giua chung luc train) - file "
          f"l4_seq_cat_*.vocab, l4_seq_char.vocab, l4_seq_numeric_norm.txt")

    dataset = StorySeqDataset(segments, cat_vocabs, char_vocab, num_norm)
    if len(dataset) < 2:
        print("[ERROR] tap train < 2 sample - khong train duoc gi ca", file=sys.stderr)
        sys.exit(1)

    # ---- MOI (LOSO) - build tap eval NEU co, dung LAI vocab/norm cua TRAIN
    # (khong fit lai) - day chinh la co che cho phep chay ca 3 fold:
    #   Fold 1 (within-host THEIA): --stories/--labels = THEIA train subset,
    #     --eval_stories/--eval_labels = THEIA held-out subset (VD tach
    #     theo story-id, xem ghi chu duoi ve cach tach).
    #   Fold 2 (within-host TRACE): tuong tu, doi sang TRACE.
    #   Fold 3 (cross-host LOSO): --stories/--labels = TOAN BO THEIA,
    #     --eval_stories/--eval_labels = TOAN BO TRACE (hoac nguoc lai) -
    #     day la fold DA BIET truoc se bi anh huong boi TRACE cmdLine null
    #     100% cho EVENT_EXECUTE (xem canh bao rieng in ra duoi day) -
    #     KHONG phai bug moi, da ghi trong Limitations tu truoc. ----
    # ---- MOI (LOSO) - build tap eval NEU co (co the gop nhieu dataset
    # giong het TRAIN o tren), dung LAI vocab/norm cua TRAIN (khong fit
    # lai) - day chinh la co che cho phep chay LOSO 3-fold:
    #   train theia_e3 + trace_e3  -> eval theia_e5
    #   train theia_e5 + trace_e3  -> eval theia_e3
    #   train theia_e5 + theia_e3  -> eval trace_e3
    # ----
    eval_dataset = None
    eval_loader = None
    if args.eval_stories:
        eval_segments = []
        for ev_stories_path, ev_labels_path in zip(args.eval_stories, args.eval_labels):
            log_ram(f"TRUOC khi xu ly EVAL {os.path.basename(ev_stories_path)}")
            eval_segments_i, eval_stats_i = stream_build_segments(
                ev_stories_path, ev_labels_path, args.max_seq_len, tag="EVAL")
            print(f"[INFO] EVAL: doc {eval_stats_i['n_stories_in_file']} story, "
                  f"{eval_stats_i['n_label_rows']} dong labels.csv tu "
                  f"{ev_stories_path} / {ev_labels_path}")
            print(f"[INFO] EVAL: {ev_stories_path} -> {len(eval_segments_i)} sample "
                  f"sau cat ({eval_stats_i['n_sids_found']}/"
                  f"{eval_stats_i['n_sids_needed']} seed_node_id can dung da tim thay)")
            eval_segments.extend(eval_segments_i)
            del eval_segments_i
            log_ram(f"sau khi xu ly xong EVAL {os.path.basename(ev_stories_path)}")

        if not eval_segments:
            print("[WARN] --eval_stories/--eval_labels khong cat duoc segment nao - "
                  "BO QUA eval, chi train (kiem tra lai seed_node_id co khop khong)",
                  file=sys.stderr)
        else:
            print(f"[INFO] TONG CONG EVAL (gop {len(args.eval_stories)} dataset): "
                  f"{len(eval_segments)} sample")
            eval_dataset = StorySeqDataset(eval_segments, cat_vocabs, char_vocab, num_norm)
            eval_label_counts = Counter(int(l) for _, labels_list in eval_segments for l in labels_list)
            print(f"[INFO] phan bo label EVAL (theo EVENT): {dict(eval_label_counts)}")

            train_set_names = set(args.stories)
            eval_set_names = set(args.eval_stories)
            if not (eval_set_names & train_set_names):
                print("[LUU Y - LOSO CROSS-HOST] --eval_stories KHONG trung file "
                      "nao voi --stories - day la fold cross-host THAT SU (dung "
                      "cho paper). NEU eval set gom TRACE E3 hoac THEIA E5, "
                      "EVENT_EXECUTE khong co cmdLine (100% rong, da xac nhan) - "
                      "model se CHI thay path_primary/secondary tu touched_"
                      "resource cho cac event nay, khong phai tu cmdline. Day "
                      "KHONG PHAI loi danh gia moi, la gioi han du lieu DA BIET "
                      "- PHAI ghi ro trong Limitations neu dung ket qua fold nay.")

    # SUA (per-event): dem label THEO TUNG EVENT (khong phai theo sample
    # nua, vi 1 sample = 1 actor co the co CA event benign LAN malicious
    # tron trong cung 1 chuoi) - flatten toan bo labels_list cua moi
    # sample de dem dung phan bo class thuc te tren tung event.
    label_counts = Counter(int(l) for _, labels_list in segments for l in labels_list)
    print(f"[INFO] phan bo label TRAIN (theo EVENT, khong phai sample): {dict(label_counts)}")
    if len(label_counts) < 2:
        print("[WARN] CHI CO 1 CLASS trong tap train - model se khong hoc "
              "duoc gi ca (loss se ve 0 ngay, do la BUG DATA khong phai bug "
              "code) - kiem tra lai labels.csv", file=sys.stderr)

    # pin_memory=True chi co ich khi dich chuyen CPU->GPU (chuyen sang
    # trang RAM "pinned" de copy nhanh hon qua PCIe) - vo hai, tu bo qua
    # neu device=cpu (torch tu canh bao neu bat pin_memory ma khong GPU,
    # nen chi bat khi thuc su dung cuda).
    use_pin = (device.type == "cuda")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_pad, pin_memory=use_pin)
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_pad,
                                  pin_memory=use_pin)

    model = L4SeqClassifier(cat_sizes, len(char_vocab.stoi)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # pos_weight cho class imbalance - TINH THEO EVENT (khong phai sample
    # nua, dung nghia voi loss per-timestep ben duoi)
    n_pos = max(label_counts.get(1, 0), 1)
    n_neg = max(label_counts.get(0, 0), 1)
    pos_weight = torch.tensor(n_neg / n_pos, device=device)
    print(f"[INFO] pos_weight = {pos_weight.item():.3f} (n_neg={n_neg}, n_pos={n_pos})")
    # SUA (per-event): reduction="none" - can loss RIENG cho tung vi tri
    # [B,T] de tu tay MASK ra padding truoc khi average, KHONG the dung
    # reduction mac dinh (se tinh ca padding vao trung binh, lam sai
    # gradient khi cac sample trong batch dai ngan khac nhau).
    # pos_weight da nam tren `device` o tren - BCEWithLogitsLoss se giu
    # nguyen device cua tensor duoc truyen vao (khong can .to(device) rieng
    # cho loss_fn vi day khong phai nn.Module co parameter can chuyen).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def run_eval_epoch():
        """MOI (LOSO) - 1 pass QUA eval_loader, KHONG backward/khong opt.step()
        (torch.no_grad()) - tra ve (avg_loss, accuracy_tho) tinh tren TUNG
        EVENT that (mask=True), dung DUNG pos_weight/loss_fn cua TRAIN (nhat
        quan, khong tinh loss khac cong thuc giua train/eval)."""
        model.eval()
        total_loss = 0.0
        total_events = 0
        total_correct = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, device)
                logit = model(batch)
                loss_per_event = loss_fn(logit, batch["labels"])
                mask_f = batch["mask"].float()
                loss = (loss_per_event * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                pred = (torch.sigmoid(logit) >= 0.5).float()
                correct = ((pred == batch["labels"]).float() * mask_f).sum()
                n_ev = int(mask_f.sum().item())
                total_loss += loss.item() * n_ev
                total_correct += correct.item()
                total_events += n_ev
        model.train()
        avg_loss = total_loss / max(total_events, 1)
        acc = total_correct / max(total_events, 1)
        return avg_loss, acc, total_events

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_events = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            opt.zero_grad()
            logit = model(batch)                      # [B, T]
            loss_per_event = loss_fn(logit, batch["labels"])  # [B, T]
            mask_f = batch["mask"].float()             # [B, T] - 1=that, 0=padding
            loss = (loss_per_event * mask_f).sum() / mask_f.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            n_ev_in_batch = int(mask_f.sum().item())
            total_loss += loss.item() * n_ev_in_batch
            total_events += n_ev_in_batch
        avg_loss = total_loss / max(total_events, 1)
        log_line = (f"[epoch {epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} "
                    f"(trung binh tren TUNG EVENT that, khong tinh padding)")
        if eval_loader is not None:
            eval_loss, eval_acc, eval_n = run_eval_epoch()
            log_line += f" | eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} (n_event={eval_n})"
        print(log_line)
        if epoch == 0 or epoch == args.epochs - 1:
            # MOI (RAM) - chi in o epoch DAU va CUOI (tranh spam log neu
            # --epochs lon), du de thay RAM co TANG DAN theo epoch khong
            # (neu co = ro ri bo nho that su trong vong lap, KHAC voi RAM
            # cao 1 lan roi giu nguyen o buoc load data phia tren).
            log_ram(f"epoch {epoch+1}/{args.epochs}")

        # MOI (KHAN CAP - Colab sap het gio, KHONG co checkpoint giua
        # chung TRUOC patch nay): LUU MODEL sau MOI epoch, GHI DE 1 file
        # DUY NHAT "l4_seq_model_checkpoint.pt" (khong phai 1 file/epoch -
        # tranh chiem disk, chi can BAN MOI NHAT). Neu Colab disconnect
        # GIUA CHUNG, file nay + vocab/norm (da luu ngay sau khi fit, xem
        # tren) la DU de load lai model dang do, KHONG mat het tien do
        # nhu truoc patch nay (truoc day CHI save() 1 LAN DUY NHAT sau
        # khi xong CA epochs LAN hnm_epochs - mat trang neu ngat som).
        # model.eval() TRUOC khi save de nhat quan voi cach export ONNX
        # cuoi cung se lam (BatchNorm/Dropout hanh xu khac train/eval),
        # roi model.train() lai NGAY de vong for epoch tiep theo dung.
        model.eval()
        torch.save(model.state_dict(),
                    os.path.join(args.out_dir, "l4_seq_model_checkpoint.pt"))
        model.train()

    # ================================================================
    # MOI - HARD NEGATIVE MINING (OHEM, in-sample) - CHI chay neu
    # --hnm_epochs > 0 (mac dinh 0 = TAT, hanh vi cu giu nguyen).
    #
    # Logic: sau khi model da hoc xong vong BINH THUONG (uniform sampling
    # tren toan bo dataset), cho model TU CHAM LAI tap TRAIN (khong phai
    # eval - OHEM chuan la mine tren chinh data da/dang dung de train,
    # KHONG phai tren held-out, nen KHONG vi pham nguyen tac tranh leakage
    # da ap dung cho eval_dataset o tren). Voi MOI sample trong dataset,
    # tinh "do kho" = max(P(malicious)) tren cac vi tri label=0 (benign)
    # CUA CHINH sample do (neu sample khong co event benign nao - toan bo
    # la malicious - do kho = 0, khong duoc tinh la "hard negative").
    #
    # Sample nao co do kho CAO (top hnm_topk_frac) = benign nhung model
    # dang predict gan 1.0 = "hard negative" that su - dung
    # WeightedRandomSampler tang xac suat duoc chon lai NHIEU LAN hon
    # trong cac epoch tiep theo (hnm_oversample_weight lan), CAC sample
    # khac (bao gom TOAN BO sample malicious) VAN giu trong tap, weight=1
    # (khong bi loai bo - tranh catastrophic forgetting nhung gi da hoc
    # dung), CHI la it duoc lap lai hon so voi hard negative.
    #
    # Loss/nhan KHONG doi gi ca (van la BCEWithLogitsLoss + pos_weight
    # nhu vong dau) - hard negative van mang dung label=0 that, model chi
    # THAY chung nhieu hon, tu dieu chinh trong so de day xac suat predict
    # cua chung xuong dung, giam False Positive that su (khac voi tang
    # pos_weight, vi tang pos_weight lam TANG false positive, khong giam).
    # ================================================================
    if args.hnm_epochs > 0:
        print(f"\n[HNM] Bat dau Hard Negative Mining: {args.hnm_epochs} epoch them, "
              f"top {args.hnm_topk_frac:.0%} sample benign kho nhat, "
              f"oversample_weight={args.hnm_oversample_weight}")

        model.eval()
        sample_difficulty = []   # (sample_idx, max_prob_tren_cac_vi_tri_label0)
        with torch.no_grad():
            scan_loader = DataLoader(dataset, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=collate_pad,
                                      pin_memory=use_pin)
            base_idx = 0
            for batch in scan_loader:
                batch = move_batch_to_device(batch, device)
                logit = model(batch)                       # [B, T]
                prob = torch.sigmoid(logit)                 # [B, T]
                mask_f = batch["mask"].float()               # [B, T]
                neg_mask = mask_f * (batch["labels"] == 0).float()   # chi vi tri
                                            # THAT (mask=1) VA label=0 (benign)
                has_neg = neg_mask.sum(dim=1) > 0            # [B] - sample nay
                                            # co it nhat 1 event benign khong
                # prob tai vi tri KHONG phai benign-that -> ep ve -1 truoc khi
                # max(), tranh vi tri malicious/padding lan vao phep max()
                masked_prob = prob * neg_mask + (-1.0) * (1.0 - neg_mask)
                max_neg_prob = masked_prob.max(dim=1).values   # [B]
                for b in range(logit.shape[0]):
                    difficulty = float(max_neg_prob[b].item()) if bool(has_neg[b]) else -1.0
                    sample_difficulty.append((base_idx + b, difficulty))
                base_idx += logit.shape[0]
        model.train()

        # loc bo sample khong co event benign nao (difficulty=-1, khong
        # phai "hard negative" - vi du sample toan bo la malicious)
        valid = [(idx, d) for idx, d in sample_difficulty if d >= 0.0]
        valid.sort(key=lambda x: x[1], reverse=True)   # kho nhat truoc
        n_hard = max(1, int(len(valid) * args.hnm_topk_frac)) if valid else 0
        hard_idx_set = {idx for idx, _ in valid[:n_hard]}

        print(f"[HNM] Tim thay {len(valid)} sample co event benign, danh dau "
              f"{len(hard_idx_set)} sample la 'hard negative' (top "
              f"{args.hnm_topk_frac:.0%}), diem kho nhat={valid[0][1]:.4f}, "
              f"diem kho thap nhat trong top={valid[n_hard-1][1]:.4f}" if valid else
              f"[HNM] [WARN] khong co sample benign nao trong tap train - BO QUA HNM")

        if hard_idx_set:
            weights = [args.hnm_oversample_weight if i in hard_idx_set else 1.0
                       for i in range(len(dataset))]
            hnm_sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                                 replacement=True)
            hnm_loader = DataLoader(dataset, batch_size=args.batch_size,
                                     sampler=hnm_sampler, collate_fn=collate_pad,
                                     pin_memory=use_pin)

            for hnm_epoch in range(args.hnm_epochs):
                total_loss = 0.0
                total_events = 0
                for batch in hnm_loader:
                    batch = move_batch_to_device(batch, device)
                    opt.zero_grad()
                    logit = model(batch)
                    loss_per_event = loss_fn(logit, batch["labels"])
                    mask_f = batch["mask"].float()
                    loss = (loss_per_event * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                    loss.backward()
                    opt.step()
                    n_ev_in_batch = int(mask_f.sum().item())
                    total_loss += loss.item() * n_ev_in_batch
                    total_events += n_ev_in_batch
                avg_loss = total_loss / max(total_events, 1)
                log_line = (f"[HNM epoch {hnm_epoch+1}/{args.hnm_epochs}] "
                            f"train_loss={avg_loss:.4f}")
                if eval_loader is not None:
                    eval_loss, eval_acc, eval_n = run_eval_epoch()
                    log_line += f" | eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} (n_event={eval_n})"
                print(log_line)

    # ---- export ----
    # MOI: dua model ve CPU TRUOC khi export ONNX - du train tren GPU,
    # ONNX graph xuat ra PHAI portable/doc lap device (C infer bang ORT
    # tren may khong chac co GPU) - export thang tu model con nam tren
    # cuda se bake device-specific behavior vao graph trong 1 so truong
    # hop va gay loi khi torch.onnx.export duyet qua model voi dummy input
    # dang nam tren CPU (dummy o duoi van tao bang torch.zeros mac dinh =
    # CPU, KHONG doi de giu portability).
    model = model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()  # tra lai VRAM ngay (khong bat buoc,
                                   # nhung tot neu chay nhieu lan trong
                                   # cung 1 session Colab tranh tich VRAM)
    model.eval()  # PHAI truoc torch.onnx.export() - BatchNorm1d trong
                    # TCN block hanh xu KHAC nhau train/eval (running
                    # mean/var vs batch stat), export luc con .train()
                    # se bake SAI hanh vi vao ONNX co dinh
    torch.save(model.state_dict(), os.path.join(args.out_dir, "l4_seq_model.pt"))
    for name, vocab in cat_vocabs.items():
        vocab.dump(os.path.join(args.out_dir, f"l4_seq_cat_{name}.vocab"))
    char_vocab.dump(os.path.join(args.out_dir, "l4_seq_char.vocab"))
    num_norm.dump(os.path.join(args.out_dir, "l4_seq_numeric_norm.txt"))

    # ONNX export - input dong truc T (seq_len), batch=1 co dinh (dong
    # bo convention l4_charcnn.c cu: C infer 1 story/lan goi, khong batch)
    T = 8  # placeholder shape cho trace - dynamic_axes se cho phep T
           # thuc te khac luc infer
    # THU TU 9 INPUT PHAI KHOP CHINH XAC voi comment l4_seq_classifier.h:
    # "cat_tag/cat_syscall/cat_errno/cat_behavior/cat_is_seed/numeric/
    # char_primary/char_secondary/mask" - cat_is_seed dung SAU
    # cat_behavior, TRUOC numeric (MOI, truoc day thieu hoan toan field
    # nay - xem lich su bug da phat hien).
    dummy = {
        "cat_tag": torch.zeros(1, T, dtype=torch.long),
        "cat_syscall": torch.zeros(1, T, dtype=torch.long),
        "cat_errno": torch.zeros(1, T, dtype=torch.long),
        "cat_behavior": torch.zeros(1, T, dtype=torch.long),
        "cat_is_seed": torch.zeros(1, T, dtype=torch.long),
        "numeric": torch.zeros(1, T, N_NUMERIC + 1, dtype=torch.float32),
        "char_primary": torch.zeros(1, T, CHAR_MAX_LEN, dtype=torch.long),
        "char_secondary": torch.zeros(1, T, CHAR_MAX_LEN, dtype=torch.long),
        "mask": torch.ones(1, T, dtype=torch.bool),
    }

    class ExportWrapper(nn.Module):
        """torch.onnx.export can input positional, khong nhan dict truc
        tiep - wrap lai thanh args ro rang theo THU TU CO DINH, C phai
        build tensor DUNG THU TU NAY (xem input_names ben duoi).

        SUA (per-event): output gio la [1, T] (1 xac suat/event, THEO
        DUNG THU TU THOI GIAN cua chuoi dua vao), KHONG PHAI [1] (1 so
        duy nhat cho ca chuoi) nhu ban truoc. **BAT BUOC BAO LAI ARCHITECT
        TRUOC KHI DUNG VOI C**: l4_infer.c hien dang doc output ONNX nhu
        1 float duy nhat (kien truc da khoa/da compile voi 5 file C patch
        truoc do) - shape moi [1,T] se can sua lai C de doc mang thay vi
        1 gia tri, KHONG tu dong tuong thich nguoc. Day la thay doi kien
        truc that su, khong phai chi doi so o Python."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, cat_tag, cat_syscall, cat_errno, cat_behavior,
                     cat_is_seed, numeric, char_primary, char_secondary, mask):
            logit = self.m({
                "cat_tag": cat_tag, "cat_syscall": cat_syscall,
                "cat_errno": cat_errno, "cat_behavior": cat_behavior,
                "cat_is_seed": cat_is_seed,
                "numeric": numeric, "char_primary": char_primary,
                "char_secondary": char_secondary, "mask": mask,
            })  # [1, T]
            return torch.sigmoid(logit)  # [1, T] - 1 xac suat/event

    wrapper = ExportWrapper(model)
    wrapper.eval()
    onnx_path = os.path.join(args.out_dir, "l4_seq_model.onnx")
    torch.onnx.export(
        wrapper,
        (dummy["cat_tag"], dummy["cat_syscall"], dummy["cat_errno"], dummy["cat_behavior"],
         dummy["cat_is_seed"], dummy["numeric"], dummy["char_primary"],
         dummy["char_secondary"], dummy["mask"]),
        onnx_path,
        input_names=["cat_tag", "cat_syscall", "cat_errno", "cat_behavior",
                     "cat_is_seed", "numeric", "char_primary", "char_secondary", "mask"],
        output_names=["prob_malicious_per_event"],  # SUA ten - phan anh
                                        # dung ban chat MOI: 1 mang, khong
                                        # phai 1 so, tranh nham lan phia C
        dynamic_axes={
            "cat_tag": {1: "seq_len"}, "cat_syscall": {1: "seq_len"},
            "cat_errno": {1: "seq_len"}, "cat_behavior": {1: "seq_len"},
            "cat_is_seed": {1: "seq_len"},
            "numeric": {1: "seq_len"}, "char_primary": {1: "seq_len"},
            "char_secondary": {1: "seq_len"}, "mask": {1: "seq_len"},
            "prob_malicious_per_event": {1: "seq_len"},  # output cung dong
                                        # truc T voi input, PHAI khai bao
                                        # dynamic o day neu khong ONNX se
                                        # bake cung T=8 (dummy) vao graph
        },
        opset_version=18,
    )
    print(f"[INFO] xong. ONNX: {onnx_path}, vocab/norm: {args.out_dir}/l4_seq_*")
    print("[WARN] NHAC LAI: neu tap train hien tai la ~20 story smoke-test, "
          "KHONG dung loss/AUC bao cao o day lam ket qua paper cuoi cung.")
    if eval_loader is None:
        print("[CANH BAO PHUONG PHAP] KHONG co --eval_stories/--eval_labels - "
              "toan bo con so loss/accuracy phia tren CHI la tren TAP TRAIN, "
              "KHONG PHAI held-out. KHONG dung so nay lam ket qua danh gia trong "
              "paper - can chay lai voi --eval_stories/--eval_labels tro vao 1 "
              "tap KHONG dung de train (within-host holdout hoac cross-host LOSO).")
    else:
        print("[LUU Y] eval_acc o tren la accuracy THO tren tung event, KHONG "
              "phan anh dung Precision/Recall thuc te khi class cuc mat can bang "
              "(~1-2% malicious) - accuracy cao gia tao (VD du doan toan 0 cung "
              "ra ~98% accuracy). Dung eval_acc CHI de theo doi model co dang hoc "
              "gi khong qua tung epoch (sanity check nhanh), KHONG dung lam con "
              "so Precision/Recall chinh thuc cho paper - cho do, chay ONNX vua "
              "xuat qua pipeline that (avro_main.c) roi do bang 19_uuid.py/"
              "20_uuid.py nhu da lam voi THEIA E3/TRACE E3, day moi la phuong "
              "phap da duoc xac nhan dung trong du an nay.")
    print("[INFO] l4_seq_classifier.c + l4_infer.c DA doc dung output ONNX "
          "[1,T] (per-event) - xac nhan lai qua project files ngay "
          "2026-08-11, khong can sua gi them ben C truoc khi chay avro_main.c "
          "voi ONNX moi nay.")


if __name__ == "__main__":
    main()
