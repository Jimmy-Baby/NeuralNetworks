#include "Matrix.h"

#include <cassert>

#include "Math.h"

#include <format>
#include <functional>

static size_t g_MatrixIdentifierCounter = 0;

Matrix::Matrix()
	: Matrix(2, 2)
{
}

Matrix::Matrix(const size_t RowCount, const size_t ColumnCount, const bool Randomise)
	: MatrixIdentifier(g_MatrixIdentifierCounter++), RowCount(RowCount), ColumnCount(ColumnCount), Data(new float[RowCount * ColumnCount])
{
	if (Data == nullptr)
	{
		throw std::runtime_error("new() failed to allocate memory for Matrix");
	}

	if (Randomise)
	{
		for (size_t Index = 0; Index < RowCount * ColumnCount; ++Index)
		{
			Data[Index] = Math::RandomFloat();
		}
	}
	else
	{
		for (size_t Index = 0; Index < RowCount * ColumnCount; ++Index)
		{
			Data[Index] = 0.0f;
		}
	}
}

Matrix::Matrix(const std::vector<std::vector<float>>& RowsOfColumns)
{
	MatrixIdentifier = g_MatrixIdentifierCounter++;

	// Verify that all rows have same count
	const size_t FirstSize = RowsOfColumns[0].size();
	for (size_t RowIndex = 0; RowIndex < RowsOfColumns.size(); ++RowIndex)
	{
		if (FirstSize != RowsOfColumns[RowIndex].size())
		{
			throw std::length_error(std::format("Not all rows have same length!"));
		}
	}

	RowCount = RowsOfColumns.size();
	ColumnCount = RowsOfColumns[0].size();
	Data = new float[RowCount * ColumnCount];

	if (Data == nullptr)
	{
		throw std::runtime_error("new() failed to allocate memory for Matrix");
	}

	// Copy the data
	for (size_t RowIndex = 0; RowIndex < RowCount; ++RowIndex)
	{
		for (size_t ColumnIndex = 0; ColumnIndex < ColumnCount; ++ColumnIndex)
		{
			Data[RowIndex * ColumnCount + ColumnIndex] = RowsOfColumns[RowIndex][ColumnIndex];
		}
	}
}

Matrix::Matrix(const Matrix& Source)
{
	MatrixIdentifier = g_MatrixIdentifierCounter++;

	RowCount = Source.RowCount;
	ColumnCount = Source.ColumnCount;

	Data = new float[RowCount * ColumnCount];

	if (Data == nullptr)
	{
		throw std::runtime_error("new() failed to allocate memory for Matrix");
	}

	// Copy data
	std::memcpy(Data, Source.Data, RowCount * ColumnCount * sizeof(float));
}

Matrix& Matrix::operator=(const Matrix& Rhs)
{
	// If matrices are same size already, just copy data and dont reallocate
	if (RowCount == Rhs.RowCount && ColumnCount == Rhs.ColumnCount)
	{
		std::memcpy(Data, Rhs.Data, RowCount * ColumnCount * sizeof(float));
		return *this;
	}

	// If not, set new size
	RowCount = Rhs.RowCount;
	ColumnCount = Rhs.ColumnCount;

	// Delete and reallocate using new size
	delete[] Data;
	Data = new float[RowCount * ColumnCount];

	// Copy data
	std::memcpy(Data, Rhs.Data, RowCount * ColumnCount * sizeof(float));

	return *this;
}

Matrix& Matrix::operator=(Matrix&& Rhs) noexcept
{
	RowCount = Rhs.RowCount;
	ColumnCount = Rhs.ColumnCount;

	Rhs.RowCount = 0;
	Rhs.ColumnCount = 0;

	Data = Rhs.Data;
	Rhs.Data = nullptr;

	return *this;
}

Matrix::~Matrix()
{
	delete[] Data;
}

Matrix Matrix::SubMatrix(const size_t RowOffset, const size_t ColumnOffset, const size_t SubMatrixRowCount, const size_t SubMatrixColumnCount) const
{
	assert(RowCount >= SubMatrixRowCount);
	assert(ColumnCount >= SubMatrixColumnCount);

	Matrix Result(SubMatrixRowCount, SubMatrixColumnCount);

	for (size_t RowIndex = 0; RowIndex < RowCount; ++RowIndex)
	{
		for (size_t ColumnIndex = 0; ColumnIndex < ColumnCount; ++ColumnIndex)
		{
			// Check starting bounds
			if (RowIndex < RowOffset || ColumnIndex < ColumnOffset)
			{
				continue;
			}

			// Check ending bounds
			if (RowIndex >= RowOffset + SubMatrixRowCount || ColumnIndex >= ColumnOffset + SubMatrixColumnCount)
			{
				continue;
			}

			const size_t SourceDataIndex = RowIndex * ColumnCount + ColumnIndex;
			const size_t DesinationRowIndex = RowIndex - RowOffset;
			const size_t DesinationColumnIndex = ColumnIndex - ColumnOffset;
			const size_t DestinationDataIndex = DesinationRowIndex * SubMatrixColumnCount + DesinationColumnIndex;

			assert(DesinationColumnIndex <= Result.GetColumnCount() * Result.GetRowCount());

			Result.Data[DestinationDataIndex] = Data[SourceDataIndex];
		}
	}

	return Result;
}

float& Matrix::At(const size_t RowIndex, const size_t ColumnIndex)
{
	return Data[RowIndex * ColumnCount + ColumnIndex];
}

float Matrix::At(const size_t RowIndex, const size_t ColumnIndex) const
{
	return Data[RowIndex * ColumnCount + ColumnIndex];
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

void Matrix::Dot(const Matrix& Rhs, Matrix& ResultOut) const
{
	assert(ColumnCount == Rhs.RowCount);
	assert(RowCount == ResultOut.RowCount);
	assert(Rhs.ColumnCount == ResultOut.ColumnCount);

	const size_t RhsColumnCount = Rhs.GetColumnCount();

	for (size_t LhsRowIndex = 0; LhsRowIndex < RowCount; ++LhsRowIndex)
	{
		for (size_t RhsColumnIndex = 0; RhsColumnIndex < RhsColumnCount; ++RhsColumnIndex)
		{
			// C = A1 * B1 + A2 * B2 + ...
			float C = 0.0f;

			for (size_t LhsColumnIndex = 0; LhsColumnIndex < ColumnCount; ++LhsColumnIndex)
			{
				const size_t RhsRowIndex = LhsColumnIndex;

				const float A = At(LhsRowIndex, LhsColumnIndex);
				const float B = Rhs.At(RhsRowIndex, RhsColumnIndex);

				C += A * B;
			}

			ResultOut.At(LhsRowIndex, RhsColumnIndex) = C;
		}
	}
}

void Matrix::Sum(const Matrix& Rhs, const Matrix& ResultOut) const
{
	assert(RowCount == Rhs.RowCount);
	assert(ColumnCount == Rhs.ColumnCount);
	assert(RowCount == ResultOut.RowCount);
	assert(ColumnCount == ResultOut.ColumnCount);

	for (size_t Index = 0; Index < RowCount * ColumnCount; ++Index)
	{
		ResultOut.Data[Index] = Data[Index] + Rhs.GetData()[Index];
	}
}

void Matrix::Sum(const Matrix& Rhs)
{
	assert(RowCount == Rhs.RowCount);
	assert(ColumnCount == Rhs.ColumnCount);

	for (size_t RowIndex = 0; RowIndex < RowCount; ++RowIndex)
	{
		for (size_t ColumnIndex = 0; ColumnIndex < ColumnCount; ++ColumnIndex)
		{
			At(RowIndex, ColumnIndex) += Rhs.At(RowIndex, ColumnIndex);
		}
	}
}

void Matrix::Activate() const
{
	for (size_t Index = 0; Index < RowCount * ColumnCount; ++Index)
	{
		Data[Index] = Math::Sigmoid(Data[Index]);
	}
}

void Matrix::PrintValues(const char* Name, const char* FormatSpecifier) const
{
	printf("%s = [\n", Name);

	for (size_t RowIndex = 0; RowIndex < RowCount; ++RowIndex)
	{
		for (size_t ColumnIndex = 0; ColumnIndex < ColumnCount; ++ColumnIndex)
		{
			char FormatString[32];
			sprintf_s(FormatString, "  %s", FormatSpecifier);
			printf(FormatString, Data[RowIndex * ColumnCount + ColumnIndex]);
		}

		printf("\n");
	}

	printf("]\n");
}

Matrix Matrix::operator*(const Matrix& Rhs) const
{
	Matrix Result(RowCount, Rhs.ColumnCount);
	Dot(Rhs, Result);
	return Result;
}

Matrix Matrix::operator+(const Matrix& Rhs) const
{
	Matrix Result(RowCount, ColumnCount);
	Sum(Rhs, Result);
	return Result;
}

Matrix& Matrix::operator+=(const Matrix& Rhs)
{
	Sum(Rhs);
	return *this;
}
