from typing import *
from numbers import Number
from collections import namedtuple
import copy

Dimensions = namedtuple('Dimensions', ['rows', 'cols'])
DimensionsType = Union[Dimensions, Tuple[int, int]]


class Matrix():

    def clone_switch(default: bool):
        def _clone_switch(in_place_function: Callable) -> Callable:
            def decorator(self, *args, in_place: bool=None, **kwargs):
                if in_place == None:
                    in_place = default
                
                if in_place:
                    return in_place_function(self, *args, **kwargs)
                
                new_matrix = Matrix(data=self.data)
                return in_place_function(new_matrix, *args, **kwargs)
            return decorator
        return _clone_switch

    def __init__(self, dimensions: DimensionsType=None, data: List[List[Number]]=None) -> None:
        
        if dimensions and data :
            raise ValueError("Either provide `dimensions` or `data`, not both.")
        
        if not dimensions and not data:
            raise ValueError("Provide a valid `dimensions` or `data`.")

        self.data = None
        self.dimensions = None

        if dimensions:

            if not isinstance(dimensions, Dimensions):
                dimensions = Dimensions(*dimensions)

            self.dimensions = dimensions
            self._create_matrix()
            self.zero()

        elif data:
            self.data = copy.deepcopy(data)
            self.dimensions = Dimensions(len(self.data), len(self.data[0]))
            self._validate()

    def _create_matrix(self) -> None:
        self.data = [[None for _ in range(self.cols)] for __ in range(self.rows)]

    def _validate(self, should_raise=True) -> bool:
        supposed_length = self.cols
        result = True
        for col in self:
            if len(col) != supposed_length:
                result = False
                
            for val in col:
                if not isinstance(val, Number):
                    result = False
                    break

            if result == False:
                if should_raise:
                    raise ValueError("Invalid matrix.")
                return False
        return True

    @clone_switch(default=True)
    def zero(self) -> None:
        for row in self.data:
            for i, val in enumerate(row):
                row[i] = 0

        return self
    
    @clone_switch(default=True)
    def _fit_new_dimensions(self, value=None) -> 'Matrix':
        '''
        If a matrix has become bigger or smaller, it will be filled up or trunctuated
        '''

        if len(self) < self.rows:
            for i in range(self.rows - len(self)):
                self.data.append([value for _ in range(self.cols) ])

        elif len(self) > self.rows:
            self.data = self.data[:self.rows]

        for i, val in enumerate(self):
            if len(val) < self.cols:
                self.data[i].extend([value for _ in range(self.cols - len(val))])
            elif len(val) > self.cols:
                self.data[i] = self.data[i][:self.cols]
        
        return self

    @clone_switch(default=True)
    def transpose(self) -> 'Matrix':
        '''
        Transpose the matrix, e.g replace all self[(i, j)] with self[(j, i)], and thus 'flip' the matrix around
        '''
        
        final_dimensions = Dimensions(self.cols, self.rows)
        x = max(self.dimensions)
        self.dimensions = Dimensions(x, x)
        self._fit_new_dimensions()

        replaced = []
        for j in range(final_dimensions.rows):
            for i in range(final_dimensions.cols):
                if (i, j) in replaced:
                    continue
                temp = self[(j, i)]
                self[(j, i)] = self[(i, j)]
                self[(i, j)] = temp
                replaced.append((j, i))
        
        self.dimensions = final_dimensions
        self._fit_new_dimensions()

        return self

    @clone_switch(default=False)
    def cumsum(self) -> 'Matrix':
        
        for i in range(self.cols):
            cumulative_sum = 0
            for j in range(self.rows):
                cumulative_sum += self[(j, i)]
                self[(j, i)] = cumulative_sum
        
        return self

    @clone_switch(default=False)
    def minormatrix(self):
        '''
        Calculate the cofactor matrix of the (square) matrix and return it 
        '''

        if not self.is_square:
            raise ValueError("Square matrix is required to compute determinant.")

        if self.rows == 1: # and logically self.dimensions.cols == 1
            return self[0][0]

        clone = Matrix(data=self.data)

        for i, row in enumerate(self.data):
            for j, val in enumerate(row):

                submatrix = Matrix(data=clone.data)
                submatrix.remove_col(i)
                submatrix.remove_row(j)

                self.data[j][i] = submatrix.determinant
        return self
    
    @clone_switch(default=False)
    def comatrix(self):
        self.minormatrix(in_place=True)

        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                self.data[j][i] *= (-1)**(i+j)
        return self
    

    @clone_switch(default=False)
    def inverse(self):
        if not self.is_square:
            raise ValueError("Square matrix is required to compute inverse matrix.")

        return (1/self.determinant) * self.adjugate(in_place=False)  

    @clone_switch(default=False)
    def adjugate(self):
        if not self.is_square:
            raise ValueError("Square matrix is required to compute adjugate matrix.")
        
        return self.comatrix(in_place=False).transpose(in_place=False)

    @clone_switch(default=True)
    def remove_row(self, idx) -> 'Matrix':
        if idx >= self.rows:
            raise IndexError("Out of bounds operation.")
        self.data.pop(idx)
        self.dimensions = Dimensions(self.rows - 1, self.cols)

        return self

    @clone_switch(default=True)
    def remove_col(self, idx) -> 'Matrix':
        if idx >= self.cols:
            raise IndexError("Out of bounds operation.")

        for i, val in enumerate(self.data):
            self.data[i].pop(idx)
        
        self.dimensions = Dimensions(self.rows, self.cols - 1)
        
        return self

    @clone_switch(default=True)
    def swap_row(self, idx1, idx2):
        
        if idx1 >= self.rows or idx2 >= self.rows:
            raise IndexError("Out of bounds operation.")

        if idx1 == idx2:
            return self

        temp = self.data[idx1]
        self.data[idx1] = list(self.data[idx2])
        self.data[idx2] = list(temp)

        return self

    @clone_switch(default=True)
    def swap_col(self, idx1, idx2):

        if idx1 >= self.cols or idx2 >= self.cols:
            raise IndexError("Out of bounds operation.")
        
        if idx1 == idx2:
            return self
        
        for i, val in enumerate(self.data):
            
            temp = self.data[i][idx1]
            self.data[i][idx1] = self.data[i][idx2]
            self.data[i][idx2] = temp
        
        return self

    

    @property
    def determinant(self) -> Number:
        '''
        Calculate the determinant of the (square) matrix and return it
        '''

        if not self.is_square:
            raise ValueError("Square matrix is required to compute determinant.")

        if self.rows == 1: # and logically self.dimensions.cols == 1
            return self[0][0]

        result = 0
        sign = 1
        for i, val in enumerate(self[0]):
            submatrix = Matrix(data=self.data[1:])
            submatrix.remove_col(i)
            result += sign * val * submatrix.determinant
            sign *= -1
        
        return result

    @property
    def is_square(self) -> bool:
        return self.rows == self.cols

    @property
    def rows(self) -> int:
        return self.dimensions.rows
    
    @property
    def cols(self) -> int:
        return self.dimensions.cols

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.dimensions != other.dimensions:
            raise ValueError("Dimension mismatch. Dimensions of matrices should be equal.")

        result = Matrix(data=self.data)

        for i, row in enumerate(result):
            for j, val in enumerate(row):
                result[(i, j)] += other[(i, j)]
        
        return result

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.dimensions != other.dimensions:
            raise ValueError("Dimension mismatch. Dimensions of matrices should be equal.")

        result = Matrix(data=self.data)

        for i, row in enumerate(result):
            for j, val in enumerate(row):
                result[(i, j)] -= other[(i, j)]
        
        return result

    def __mul__(self, other: Union['Matrix', Number]):
        
        if isinstance(other, Number):
            result = Matrix(data=self.data)
            for i in range(result.rows):
                for j in range(result.cols):
                    result[(i, j)] *= other
                    if result[(i, j)].is_integer():
                        result[(i, j)] = int(result[(i, j)])
        else:
            if self.cols != other.rows:
                raise ValueError("Dimension mismatch. To compute matrix C = A * B, The amount of columns of matrix A should equal the amount of rows of matrix B.")
            result = Matrix(dimensions=Dimensions(self.rows, other.cols))

            for i, row in enumerate(result):
                for j, val in enumerate(row):
                    
                    multiplication = float(sum(self[(i, k)] * other[(k, j)] for k in range(self.cols)))
                    if multiplication.is_integer():
                        multiplication = int(multiplication)

                    result[(i, j)] = multiplication
        return result
    
    def __rmul__(self, other: Union['Matrix', Number]):
        if isinstance(other, Number):
            return self * other
        else:
            return Matrix.__mul__(other, self)

    def __getitem__(self, key: Union[DimensionsType, int]) -> Union[Tuple[Number], Number]:
        
        if isinstance(key, tuple):
            key = Dimensions(*key)

        if isinstance(key, Dimensions):
            return self.data[key.rows][key.cols]
        else:
            return tuple(self.data[key])
    
    def __setitem__(self, key: Union[DimensionsType, int], value: Union[int, List[Number]]) -> None:
        
        if isinstance(key, tuple):
            key = Dimensions(*key)

        if isinstance(key, Dimensions):
            if key.rows >= self.rows or key.cols >= self.cols:
                raise IndexError("Out of bounds operation.")
            self.data[key.rows][key.cols] = value
        else:

            if not isinstance(value, list):
                raise TypeError("Column type mismatch")
            
            if key >= self.rows:
                raise IndexError("Out of bounds operation")
            
            if len(value) > self.cols:
                raise IndexError("Out of bounds operation")

            self.data[key] = value
            self._validate()

    def __iter__(self):
        yield from self.data

    def __str__(self):
        result = "\n"
        for row in self:
            result += "[ "
            for val in row:
                result += str(val) + ' '
            result += "]\n"
        return result

    def __repr__(self):
        return f"Matrix(data={repr(self.data)})"
    
    def __len__(self):
        return len([_ for _ in self])

m = Matrix(data=[
    [3, 0, 2],
    [2, 0, -2],
    [0, 1, 1]
])

print(m.inverse())