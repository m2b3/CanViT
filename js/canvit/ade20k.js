// The standard ADE20K palette: 150 RGB colors indexed by class, copied from
// mmsegmentation's ADE20KDataset.METAINFO["palette"] (mmseg/datasets/ade.py).
// Class names come from each bundle's manifest (readout.class_names).

import { hexBytes } from "./hex.js";

export const ADE20K_PALETTE = hexBytes(
  "787878b4787806e6e650323204c8037878508c8c8ccc05ffe6e6e604fa07e005ffebff0796053d78784608ff33ff0652" +
  "8fff8cccff04ff3307cc46030066c83de6faff06330b66ffff0747ff09e00907e6dcdcdcff095c7009ff08ffd607ffe0" +
  "ffb8060aff47ff290a07ffffe0ff086608ffff3d06ffc207ff7a0800ff14ff0829ff05990633ffeb0cffa0961400a3ff" +
  "8c8c8cfa0a0f14ff001fff00ff1f00ffe00099ff000000ffff470000ebff00adff1f00ff0bc8c8ff520000fff5003dff" +
  "00ff7000ff85ff0000ffa300ff6600c2ff00008fff33ff000052ff00ff2900ffad0a00ffadff0000ff99ff5c00ff00ff" +
  "ff00f5ff0066ffad00ff0014ffb8b8001fff00ff3d0047ffff00cc00ffc200ff52000aff0070ff3300ff00c2ff007aff" +
  "00ffa3ff990000ff0aff70008fff005200ffa3ff00ffeb0008b8aa8500ff00ff5cb800ffff001f00b8ff00d6ffff0070" +
  "5cff0000e0ff70e0ff46b8a0a300ff9900ff47ff00ff00a3ffcc00ff008f00ffeb85ff00ff00ebf500ffff007afff500" +
  "0abed4d6ff0000ccff1400ffffff000099ff0029ff00ffcc2900ff29ff00ad00ff00f5ff4700ff7a00ff00ffb8005cff" +
  "b8ff000085ffffd60019c2c266ff005c00ff");
