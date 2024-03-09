#include <cstdio>
#include <cstdlib>
#include <format>

#include "PatternGradientDescent.h"
#include "../Math.h"

namespace PatternGradDesc
{
	double g_TrainingData[][2] =
	{
		{ 0.000000, 000.000000 },
		{ 1.000000, 050.000000 },
		{ 2.000000, 100.000000 },
		{ 3.000000, 150.000000 },
		{ 4.000000, 200.000000 },
	};

	double CalculateCost(const double Weight)
	{
		double Result = 0.0;

		constexpr size_t TrainingDataSize = std::size(g_TrainingData);

		for (size_t Index = 0; Index < TrainingDataSize; ++Index)
		{
			// Get the current training sample
			const double TrainingSampleX = g_TrainingData[Index][0];

			// Use the model to evaluate an output
			const double ModelPrediction = TrainingSampleX * Weight;

			// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
			const double PredictionError = ModelPrediction - g_TrainingData[Index][1];

			// Square the error to get a positive number, and add it to the result
			Result += PredictionError * PredictionError;
		}

		// Average out all the accumulated error
		Result /= TrainingDataSize;

		return Result;
	}

	double CalculateGradientDescentDistance(const double Weight)
	{
		double Result = 0.0;

		constexpr size_t TrainingDataSize = std::size(g_TrainingData);

		for (size_t Index = 0; Index < TrainingDataSize; ++Index)
		{
			const double TrainingSample = g_TrainingData[Index][0];
			const double ExpectedResult = g_TrainingData[Index][1];

			const double ModelPrediction = TrainingSample * Weight;
			const double ModelError = ModelPrediction - ExpectedResult;

			Result += 2 * ModelError * TrainingSample;
		}

		// Average out all of the gradient descent results
		Result /= TrainingDataSize;

		return Result;
	}

	// Actual formula:
	// y = x * 50
	//
	// Model formula:
	// y = x * w

	void Run()
	{
		// Random double to use in our starting point for our weight
		//constexpr double Randomdouble = 0.01f;
		const double Randomdouble = Math::RandomFloat();

		// Calculate our starting weight
		double Weight = Randomdouble * 10.0f;

		printf("Cost = %.15f, Weight = %.15f\n", CalculateCost(Weight), Weight);

		for (size_t Index = 0; Index < 1000; ++Index)
		{
			constexpr double Rate = 1e-2f;

			// Calculate the current average distance from correct weights
			const double Distance = CalculateGradientDescentDistance(Weight);

			// Adjust the weight
			Weight -= Rate * Distance;

			// Calculate the new cost after adjusting
			const double NewCost = CalculateCost(Weight);

			// Print results of this iteration
			printf("Cost = %.15f, Weight = %.15f\n", NewCost, Weight);
			
			if (std::format("{:.15f}", NewCost) == "0.000000000000000")
			{
				printf("Breaking!\n");
				break;
			}
		}

		printf("------------------------------\n");
		printf("Final weight = %.15f\n", Weight);
		printf("------------------------------\n");
		printf("Results:\n");

		for (int Index = 0; Index < 10; ++Index)
		{
			const double X = Index + 1;
			const double W = Weight;
			const double Y = X * W;

			printf("{ %09.15f, %010.15f }\n", X, Y);
		}
	}
}
