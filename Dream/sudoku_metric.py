import numpy as np

def is_valid_sudoku(input, prediction):
    ### check 4x4 sudoku result

    prediction = prediction[:len(input)]
    input_array = np.array([list(map(int, row)) for row in input.strip().split('\n')])
    try:
        grid = np.array([list(map(int, row)) for row in prediction.strip().split('\n')])
        if grid.shape != (4,4):
            return False
    except:
        return False
    
    # Create a mask for non-zero positions in the input
    non_zero_mask = input_array != 0
    
    # Check if the non-zero positions in the input match the output
    if not np.all(input_array[non_zero_mask] == grid[non_zero_mask]):
        return False

    # Check if each row, column, and subgrid contains the digits 1 to 4
    expected_set = {1, 2, 3, 4}
    
    # Check rows
    for row in grid:
        if set(row) != expected_set:
            return False
    
    # Check columns
    for col in range(4):
        if set(grid[row][col] for row in range(4)) != expected_set:
            return False
    
    # Check 2x2 subgrids
    for start_row in (0, 2):
        for start_col in (0, 2):
            subgrid = {grid[r][c] for r in range(start_row, start_row + 2) for c in range(start_col, start_col + 2)}
            if subgrid != expected_set:
                return False

    return True

if __name__ == "__main__":
    print(is_valid_sudoku("1000\n0002\n4003\n0000", "1234\n3412\n4123\n2341"))
    print(is_valid_sudoku("0020\n0034\n0400\n1000", "2413\n3142\n4321\n1234"))