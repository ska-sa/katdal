import argparse
from abc import ABC, abstractmethod
import os.path
import pickle
import re
import tempfile

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq


class EncodingStrategy(ABC):
  @abstractmethod
  def encode(self) -> pa.array:
    raise NotImplementedError

class BaseEncodingStrategy(EncodingStrategy):
  def __init__(self, key: str, data: npt.NDArray):
    self.key = key
    self.data = data

class NumericDataStrategy(BaseEncodingStrategy):
  """ Encode numeric numpy arrays"""
  def encode(self)  -> pa.array:
    # Encode complex arrays as float arrays with an extra dimension
    if np.iscomplexobj(self.data):
      data = self.data.view(self.data.real.dtype).reshape(self.data.shape + (2,))
    else:
      data = self.data

    if data.ndim > 1:
      data = pa.FixedShapeTensorArray.from_numpy_ndarray(data)

    return pa.array(data)

class ObjectStrategy(BaseEncodingStrategy):
  """ Encode object arrays, preferably as categoricals with pickled object values """
  def encode(self) -> pa.array:
    try:
      values = {}
      indices = []
      for o in self.data:
        indices.append(values.setdefault(o, len(values)))
    except TypeError:
      print(f"{self.key} not hashable, duplicates may be pickled")
      return [pickle.dumps(o) for o in self.data]
    else:
      return pa.DictionaryArray.from_arrays(indices, [pickle.dumps(o) for o in values])

class BpCalStrategy(BaseEncodingStrategy):
  """ Encode large bpcal solution arrays by storing individual dump values as categorical raw bytes"""
  def encode(self) -> pa.array:
    try:
      values = {}
      indices = []
      for o in self.data:
        i, _ = values.setdefault(o.tobytes(), (len(values), o))
        indices.append(i)
    except TypeError:
      return NumericDataStrategy(self.key, self.data).encode()
    else:
      dict_array = [pickle.dumps(o) for _, o in values.values()]
      return pa.DictionaryArray.from_arrays(indices, dict_array)

BPCAL_REGEX = re.compile(r"^.*cal_product_B.*$")
KEY_STRATEGY = {BPCAL_REGEX: BpCalStrategy}

def encoding_strategy(key: str, data: npt.NDArray) -> EncodingStrategy:
  """ Return an encoding strategy dependent on the key and data"""
  for regex, strategy_cls in KEY_STRATEGY.items():
    if regex.match(key):
      return strategy_cls(key, data)

  if data.dtype == object:
    return ObjectStrategy(key, data)

  return NumericDataStrategy(key, data)

def parquet_export(telstate, sensors):
  """
  Exports a katdal SensorCache to parquet as well as other smaller attrs.

  Args:
    telstate: katdal telstate
    sensors: katdal SensorCache
  """
  IGNORED = {"sdisp_custom_signals"}

  sensor_keys = (k for k in sensors.keys() if k not in IGNORED)
  table_data = {k: encoding_strategy(k, sensors[k]).encode() for k in sensor_keys}
  table = pa.Table.from_pydict(table_data)
  cbid = telstate["capture_block_id"]
  table_name = f"{tempfile.gettempdir()}{os.path.sep}telstate-{cbid}.parquet"
  pq.write_table(table, table_name, row_group_size=10, compression="zstd")
  print(f"Wrote to {table_name}")

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("parquet")
  args = p.parse_args()

  # 1. Read the metadata from the file
  parquet_file = pq.ParquetFile(args.parquet)
  metadata = parquet_file.metadata

  column_sizes = {}
  total_file_size = 0

  # 2. Iterate through all row groups
  for i in range(metadata.num_row_groups):
      row_group_meta = metadata.row_group(i)
      total_file_size += row_group_meta.total_byte_size

      # 3. Iterate through all column chunks in the row group
      for j in range(row_group_meta.num_columns):
          column_chunk_meta = row_group_meta.column(j)
          column_name = column_chunk_meta.path_in_schema

          # Use the compressed size for the file size contribution
          current_sizes = column_sizes.setdefault(column_name, [0, 0])
          current_sizes[0] += column_chunk_meta.total_uncompressed_size
          current_sizes[1] += column_chunk_meta.total_compressed_size


  # 4. Calculate relative size
  print("Column | Compressed Size (Bytes) | Relative Size (%) | Uncompressed Size (Bytes) | Relative Size (%)")
  print("-------|-------------------------|------------------------------------------------------------------")
  for column, (full, compressed) in sorted(column_sizes.items(), key=lambda k: k[1][0], reverse=True):
      full_relative = (full / total_file_size) * 100 if total_file_size > 0 else 0
      compressed_relative = (compressed / total_file_size) * 100 if total_file_size > 0 else 0
      print(f"{column:55s} | {compressed:12,} | {compressed_relative:6.2f} | {full:12,} | {full_relative:6.2f}")