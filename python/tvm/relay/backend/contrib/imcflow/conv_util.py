import numpy as np

class ConvUtil:
  def __init__(self, iH, iW, padding, stride, kH, kW):
    self.iH = iH
    self.iW = iW
    self.padding = padding
    self.stride = stride
    self.kH = kH
    self.kW = kW
    self.oH = (iH - kH + 2 * padding) // stride + 1
    self.oW = (iW - kW + 2 * padding) // stride + 1
    self.padded_width = iW + 2 * padding
    self.padded_height = iH + 2 * padding
    self.current_row = 0
    self.current_col = 0

  def is_padding_location(self, row, col):
    return (
      row < self.padding or
      row >= (self.iH + self.padding) or
      col < self.padding or
      col >= (self.iW + self.padding)
    )

  def update_coordinates(self):
    self.current_col += 1  # Increment column
    if self.current_col >= self.padded_width:
      self.current_col = 0  # Reset column
      self.current_row += 1  # Increment row

    if self.current_row >= self.padded_height:
      self.current_row, self.current_col = 0, 0  # Reset both

  def compute_pixel_read_count(self, target_row, target_col):
    read_count = 0
    while not (target_row == self.current_row and target_col == self.current_col):
      if not self.is_padding_location(self.current_row, self.current_col):
        read_count += 1

      self.update_coordinates()

    if not self.is_padding_location(self.current_row, self.current_col):
      read_count += 1

    self.update_coordinates()

    return read_count

  def calculate_input_read_counts(self):
    read_counts = np.zeros((self.oH, self.oW), dtype=np.int32)

    for out_row in range(self.oH):
      for out_col in range(self.oW):
        target_row = out_row * self.stride + self.kH - 1
        target_col = out_col * self.stride + self.kW - 1

        read_counts[out_row, out_col] = self.compute_pixel_read_count(target_row, target_col)

    return read_counts

  def extract_patterns(self, matrix):
    groups = []
    current_pattern = matrix[0]
    count = 0

    for row in matrix:
      if not np.array_equal(row, current_pattern):
        groups.append(self._create_group(current_pattern, count))
        current_pattern = row
        count = 0
      count += 1

    groups.append(self._create_group(current_pattern, count))
    return groups

  def _create_group(self, pattern, count):
    return {
      "pattern": pattern,
      "count": count
    }

  def get_convolution_patterns(self):
    read_count_matrix = self.calculate_input_read_counts()
    return self.extract_patterns(read_count_matrix)

  def extract_row_patterns(self, row_pattern):
    return self.extract_patterns(np.array(row_pattern))

  def extract_2d_patterns(self):
    # Extract row-level patterns
    row_patterns = self.get_convolution_patterns()
    all_2d_patterns = []

    for row_group in row_patterns:
      col_patterns = self.extract_patterns(row_group["pattern"])
      all_2d_patterns.append({
        "count": row_group["count"],
        "patterns": col_patterns
      })

    return all_2d_patterns

# Test code
if __name__ == "__main__":
  utils = ConvUtil(8, 8, 2, 2, 3, 3)
  print("Input Read Count Matrix:")
  print(utils.calculate_input_read_counts())

  print("Convolution Patterns:")
  print(utils.get_convolution_patterns())

  print("2D Patterns:")
  print(utils.extract_2d_patterns())
