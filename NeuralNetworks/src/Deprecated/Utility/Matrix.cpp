// Precompiled headers
#include "Pch.h"

#include "Matrix.h"
#include "Math.h"

size_t Matrix::MatrixIdCounter = 0;

Matrix::Matrix()
	: Matrix(2, 2)
{
}

Matrix::Matrix(const size_t rowCount, const size_t columnCount, const bool randomise)
	: MatrixIdentifier(MatrixIdCounter++), RowCount(rowCount), ColumnCount(columnCount), Data(new float[rowCount * columnCount])
{
	const size_t totalElements = rowCount * columnCount;
	
	if (randomise)
	{
		for (size_t index = 0; index < totalElements; ++index)
		{
			Data[index] = Math::RandomFloat();
		}
	}
	else
	{
		std::memset(Data, 0, totalElements * sizeof(float));
	}
}

Matrix::Matrix(const std::vector<std::vector<float>>& rowsOfColumns)
{
	MatrixIdentifier = MatrixIdCounter++;

	// Verify input is not empty
	if (rowsOfColumns.empty() || rowsOfColumns[0].empty())
	{
		throw std::invalid_argument("Cannot create matrix from empty data");
	}

	// Verify that all rows have same count
	const size_t firstRowSize = rowsOfColumns[0].size();
	for (const auto& row : rowsOfColumns)
	{
		if (firstRowSize != row.size())
		{
			throw std::length_error(std::format("Not all rows have same length!"));
		}
	}

	RowCount = rowsOfColumns.size();
	ColumnCount = rowsOfColumns[0].size();
	Data = new float[RowCount * ColumnCount];

	// Copy the data (prioritise improved cache locality by iterating in row-major order
	for (size_t rowIndex = 0; rowIndex < RowCount; ++rowIndex)
	{
		const auto& row = rowsOfColumns[rowIndex];
		const size_t rowOffset = rowIndex * ColumnCount;
		for (size_t columnIndex = 0; columnIndex < ColumnCount; ++columnIndex)
		{
			Data[rowOffset + columnIndex] = row[columnIndex];
		}
	}
}

Matrix::Matrix(const Matrix& other)
	: MatrixIdentifier(MatrixIdCounter++), RowCount(other.RowCount), ColumnCount(other.ColumnCount), Data(new float[other.RowCount * other.ColumnCount])
{
	// Copy data
	std::memcpy(Data, other.Data, RowCount * ColumnCount * sizeof(float));
}

Matrix::Matrix(Matrix&& other) noexcept
	: MatrixIdentifier(MatrixIdCounter++), RowCount(other.RowCount), ColumnCount(other.ColumnCount), Data(other.Data)
{
	other.RowCount = 0;
	other.ColumnCount = 0;
	other.Data = nullptr;
}

Matrix& Matrix::operator=(const Matrix& rhs)
{
	if (this == &rhs)
	{
		return *this;
	}

	// If matrices are same size already, just copy data and dont reallocate
	if (RowCount == rhs.RowCount && ColumnCount == rhs.ColumnCount)
	{
		std::memcpy(Data, rhs.Data, RowCount * ColumnCount * sizeof(float));
		return *this;
	}

	// If not, set new size
	RowCount = rhs.RowCount;
	ColumnCount = rhs.ColumnCount;

	// Delete and reallocate using new size
	delete[] Data;
	Data = new float[RowCount * ColumnCount];

	// Copy data
	std::memcpy(Data, rhs.Data, RowCount * ColumnCount * sizeof(float));

	return *this;
}

Matrix& Matrix::operator=(Matrix&& rhs) noexcept
{
	// Handle self-assignment
	if (this == &rhs)
	{
		return *this;
	}

	// Delete existing data
	delete[] Data;

	RowCount = rhs.RowCount;
	ColumnCount = rhs.ColumnCount;
	Data = rhs.Data;
	
	rhs.RowCount = 0;
	rhs.ColumnCount = 0;
	rhs.Data = nullptr;

	return *this;
}

Matrix::~Matrix()
{
	delete[] Data;
}

Matrix Matrix::SubMatrix(const size_t rowOffset, const size_t columnOffset, const size_t subMatrixRowCount, const size_t subMatrixColumnCount) const
{
	assert(rowOffset + subMatrixRowCount <= RowCount);
	assert(columnOffset + subMatrixColumnCount <= ColumnCount);

	Matrix result(subMatrixRowCount, subMatrixColumnCount);

	// Only iterate over the submatrix region, not the entire matrix
	for (size_t rowIndex = 0; rowIndex < subMatrixRowCount; ++rowIndex)
	{
		const size_t sourceRowIndex = rowOffset + rowIndex;
		const size_t sourceRowOffset = sourceRowIndex * ColumnCount + columnOffset;
		const size_t destRowOffset = rowIndex * subMatrixColumnCount;
		
		// Copy entire row segment in one operation for better cache performance
		std::memcpy(&result.Data[destRowOffset], &Data[sourceRowOffset], subMatrixColumnCount * sizeof(float));
	}

	return result;
}

float& Matrix::At(const size_t rowIndex, const size_t columnIndex)
{
	assert(rowIndex < RowCount && "Row index out of bounds");
	assert(columnIndex < ColumnCount && "Column index out of bounds");
	return Data[rowIndex * ColumnCount + columnIndex];
}

float Matrix::At(const size_t rowIndex, const size_t columnIndex) const
{
	assert(rowIndex < RowCount && "Row index out of bounds");
	assert(columnIndex < ColumnCount && "Column index out of bounds");
	return Data[rowIndex * ColumnCount + columnIndex];
}

float* Matrix::GetData() const
{
	return Data;
}

size_t Matrix::GetRowCount() const
{
	return RowCount;
}

size_t Matrix::GetColumnCount() const
{
	return ColumnCount;
}

void Matrix::Dot(const Matrix& rhs, Matrix& resultOut) const
{
	assert(ColumnCount == rhs.RowCount && "Matrix dimensions incompatible for multiplication");
	assert(RowCount == resultOut.RowCount && "Result matrix row count mismatch");
	assert(rhs.ColumnCount == resultOut.ColumnCount && "Result matrix column count mismatch");

	const size_t rhsColumnCount = rhs.GetColumnCount();
	const float* rhsData = rhs.GetData();
	float* resultData = resultOut.GetData();

	// Optimised matrix multiplication with better cache locality
	// Zero out result first
	std::memset(resultData, 0, RowCount * rhsColumnCount * sizeof(float));

	for (size_t lhsRowIndex = 0; lhsRowIndex < RowCount; ++lhsRowIndex)
	{
		const size_t lhsRowOffset = lhsRowIndex * ColumnCount;
		const size_t resultRowOffset = lhsRowIndex * rhsColumnCount;
		
		for (size_t lhsColumnIndex = 0; lhsColumnIndex < ColumnCount; ++lhsColumnIndex)
		{
			const float a = Data[lhsRowOffset + lhsColumnIndex];
			const size_t rhsRowOffset = lhsColumnIndex * rhsColumnCount;
			
			// Vectorizable inner loop - better for compiler optimization
			for (size_t rhsColumnIndex = 0; rhsColumnIndex < rhsColumnCount; ++rhsColumnIndex)
			{
				resultData[resultRowOffset + rhsColumnIndex] += a * rhsData[rhsRowOffset + rhsColumnIndex];
			}
		}
	}
}

void Matrix::Sum(const Matrix& rhs, Matrix& resultOut) const
{
	assert(RowCount == rhs.RowCount && "Matrix row count mismatch for addition");
	assert(ColumnCount == rhs.ColumnCount && "Matrix column count mismatch for addition");
	assert(RowCount == resultOut.RowCount && "Result matrix row count mismatch");
	assert(ColumnCount == resultOut.ColumnCount && "Result matrix column count mismatch");

	const size_t totalElements = RowCount * ColumnCount;
	const float* rhsData = rhs.GetData();
	float* resultData = resultOut.GetData();
	
	// Vectorisable loop for better performance
	for (size_t index = 0; index < totalElements; ++index)
	{
		resultData[index] = Data[index] + rhsData[index];
	}
}

void Matrix::Sum(const Matrix& rhs)
{
	assert(RowCount == rhs.RowCount && "Matrix row count mismatch for addition");
	assert(ColumnCount == rhs.ColumnCount && "Matrix column count mismatch for addition");

	const size_t totalElements = RowCount * ColumnCount;
	const float* rhsData = rhs.GetData();
	
	// Vectorizable loop for better performance
	for (size_t index = 0; index < totalElements; ++index)
	{
		Data[index] += rhsData[index];
	}
}

void Matrix::Activate()
{
	const size_t totalElements = RowCount * ColumnCount;
	
	for (size_t index = 0; index < totalElements; ++index)
	{
		Data[index] = Math::Sigmoid(Data[index]);
	}
}

void Matrix::PrintValues(const char* name, const char* formatSpecifier) const
{
	printf("%s = [\n", name);

	for (size_t rowIndex = 0; rowIndex < RowCount; ++rowIndex)
	{
		const size_t rowOffset = rowIndex * ColumnCount;
		for (size_t columnIndex = 0; columnIndex < ColumnCount; ++columnIndex)
		{
			char formatString[32];
			sprintf_s(formatString, "  %s", formatSpecifier);
			printf(formatString, Data[rowOffset + columnIndex]);
		}

		printf("\n");
	}

	printf("]\n");
}

Matrix Matrix::operator*(const Matrix& rhs) const
{
	assert(ColumnCount == rhs.RowCount && "Matrix dimensions incompatible for multiplication");
	
	Matrix result(RowCount, rhs.ColumnCount);
	Dot(rhs, result);
	return result;
}

Matrix Matrix::operator+(const Matrix& rhs) const
{
	assert(RowCount == rhs.RowCount && ColumnCount == rhs.ColumnCount && "Matrix dimensions must match for addition");
	
	Matrix result(RowCount, ColumnCount);
	Sum(rhs, result);
	return result;
}

Matrix& Matrix::operator+=(const Matrix& rhs)
{
	Sum(rhs);
	return *this;
}
