#include "ModularNN.h"

#include "Layer.h"
#include "OutputLayer.h"

ModularNn::ModularNn()
	: Output_(std::make_unique<OutputLayer>())
{
}

ModularNn::~ModularNn()
{
}

size_t ModularNn::LayerCount() const
{
	return HiddenLayers_.size();
}

const std::vector<std::unique_ptr<Layer>>& ModularNn::GetLayers() const
{
	return HiddenLayers_;
}

void ModularNn::Initialise(const TScalar& Mean, const TScalar& StandardDeviation)
{
	VerifyNetwork();

	for (size_t Index = 0; Index < LayerCount(); Index++)
	{
		HiddenLayers_[Index]->Initialise(Mean, StandardDeviation);
	}
}

void ModularNn::Forward(const TMatrix& Input)
{
	const int NumberOfLayers = LayerCount();

	if (NumberOfLayers <= 0)
	{
		return;
	}

	if (Input.rows() != HiddenLayers_[0]->InputsCount())
	{
		throw std::invalid_argument("Input has incorrect size");
	}

	// Forward to first layer
	HiddenLayers_[0]->Forward(Input);

	// Recursively forward data through network
	for (size_t Index = 1; Index < NumberOfLayers; Index++)
	{
		HiddenLayers_[Index]->Forward(HiddenLayers_[Index - 1]->Output());
	}
}

void ModularNn::Update(Wiggler& WigglerObject)
{
	const int NumberOfLayers = LayerCount();

	if (NumberOfLayers <= 0)
	{
		return;
	}

	for (size_t Index = 0; Index < NumberOfLayers; Index++)
	{
		HiddenLayers_[Index]->Update(WigglerObject);
	}
}

///
/// Fit the model based on the given data
///
/// \param opt        An object that inherits from the Optimizer class, indicating the optimization algorithm to use.
/// \param x          The predictors. Each column is an observation.
/// \param y          The response variable. Each column is an observation.
/// \param batch_size Mini-batch size.
/// \param epoch      Number of epochs of training.
/// \param seed       Set the random seed of the %RNG if `seed > 0`, otherwise
///                   use the current random state.
///
template <typename DerivedX, typename DerivedY>
bool fit(Optimizer& opt, const Eigen::MatrixBase<DerivedX>& x,
         const Eigen::MatrixBase<DerivedY>& y,
         int batch_size, int epoch, int seed = -1)
{
	// We do not directly use PlainObjectX since it may be row-majored if x is passed as mat.transpose()
	// We want to force XType and YType to be column-majored
	using PlainObjectX = typename Eigen::MatrixBase<DerivedX>::PlainObject;
	using PlainObjectY = typename Eigen::MatrixBase<DerivedY>::PlainObject;
	using XType = Eigen::Matrix<typename PlainObjectX::Scalar, PlainObjectX::RowsAtCompileTime, PlainObjectX::ColsAtCompileTime>;
	using YType = Eigen::Matrix<typename PlainObjectY::Scalar, PlainObjectY::RowsAtCompileTime, PlainObjectY::ColsAtCompileTime>;
	const int nlayer = num_layers();

	if (nlayer <= 0)
	{
		return false;
	}

	// Reset optimizer
	opt.reset();

	// Create shuffled mini-batches
	if (seed > 0)
	{
		m_rng.seed(seed);
	}

	std::vector<XType> x_batches;
	std::vector<YType> y_batches;
	const int nbatch = internal::create_shuffled_batches(x, y, batch_size, m_rng,
	                                                     x_batches, y_batches);
	// Set up callback parameters
	m_callback->m_nbatch = nbatch;
	m_callback->m_nepoch = epoch;

	// Iterations on the whole data set
	for (int k = 0; k < epoch; k++)
	{
		m_callback->m_epoch_id = k;

		// Train on each mini-batch
		for (int i = 0; i < nbatch; i++)
		{
			m_callback->m_batch_id = i;
			m_callback->pre_training_batch(this, x_batches[i], y_batches[i]);
			this->forward(x_batches[i]);
			this->backprop(x_batches[i], y_batches[i]);
			this->update(opt);
			m_callback->post_training_batch(this, x_batches[i], y_batches[i]);
		}
	}

	return true;
}

///
/// Use the fitted model to make predictions
///
/// \param x The predictors. Each column is an observation.
///
Matrix predict(const Matrix& x)
{
	const int nlayer = num_layers();

	if (nlayer <= 0)
	{
		return Matrix();
	}

	this->forward(x);
	return m_layers[nlayer - 1]->output();
}

void ModularNn::AddLayer(std::unique_ptr<Layer>&& Layer)
{
	HiddenLayers_.emplace_back(std::move(Layer));
}

void ModularNn::SetOutput(std::unique_ptr<OutputLayer>&& Output)
{
	Output_ = std::move(Output);
}

void ModularNn::VerifyNetwork() const
{
	const int NumberOfLayers = LayerCount();

	if (NumberOfLayers <= 1)
	{
		return;
	}

	for (size_t Index = 1; Index < NumberOfLayers; Index++)
	{
		if (HiddenLayers_[Index]->InputsCount() != HiddenLayers_[Index - 1]->OutputsCount())
		{
			throw std::invalid_argument("Layer inputs/outputs do not match");
		}
	}
}
