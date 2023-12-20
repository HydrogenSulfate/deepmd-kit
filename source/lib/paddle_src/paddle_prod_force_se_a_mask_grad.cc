#include "paddle/extension.h"

REGISTER_OP(prod_force_se_a_mask_grad)
    .Attr("T: {float32, float64} = float32")
    .Input("grad: T")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("mask: int32")
    .Input("nlist: int32")
    .Attr("total_atom_num: int")
    .Output("grad_net: T");

template <typename T>
void ProdForceSeAMaskGradOpCPUKernel(int nframes,
                                     int total_atom_num,
                                     const T* grad,
                                     const T* net_deriv,
                                     const T* in_deriv,
                                     const int* mask,
                                     const int* nlist,
                                     T* grad_net) {
  int nloc = total_atom_num;
  int ndescrpt = nloc > 0 ? nframes * nloc * 3 / net_deriv.shape()[1] : 0;
  int nnei = total_atom_num;

#pragma omp parallel for
  for (int kk = 0; kk < nframes; ++kk) {
    int grad_iter = kk * nloc * 3;
    int net_iter = kk * nloc * ndescrpt;
    int in_iter = kk * nloc * ndescrpt * 3;
    int nlist_iter = kk * nloc * nnei;
    int mask_iter = kk * nloc;
    int grad_net_iter = kk * nloc * ndescrpt;

    // reset the frame to 0
    for (int ii = 0; ii < nloc; ++ii) {
      for (int aa = 0; aa < ndescrpt; ++aa) {
        grad_net[grad_net_iter + ii * ndescrpt + aa] = 0.0;
      }
    }

    // compute grad of one frame
    for (int ii = 0; ii < nloc; ++ii) {
      int i_idx = ii;

      // deriv wrt center atom
      for (int aa = 0; aa < ndescrpt; ++aa) {
        for (int dd = 0; dd < 3; ++dd) {
          grad_net[grad_net_iter + i_idx * ndescrpt + aa] -=
              grad[grad_iter + i_idx * 3 + dd] *
              in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + dd];
        }
      }

      // loop over neighbors
      for (int jj = 0; jj < nnei; ++jj) {
        int j_idx = nlist[nlist_iter + i_idx * nnei + jj];
        // Check if atom j_idx is virtual or if the i_idx is virtual.
        if (j_idx == i_idx || j_idx < 0) {
          continue;
        }
        /*
        if (j_idx > nloc)
          j_idx = j_idx % nloc;
        if (j_idx < 0)
          continue;
        */
        int aa_start, aa_end;
        aa_start = jj * 4;
        aa_end = jj * 4 + 4;
        // make_descript_range (aa_start, aa_end, jj);
        for (int aa = aa_start; aa < aa_end; ++aa) {
          for (int dd = 0; dd < 3; ++dd) {
            grad_net[grad_net_iter + i_idx * ndescrpt + aa] +=
                grad[grad_iter + j_idx * 3 + dd] *
                in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + dd];
          }
        }
      }
    }
  }
}

std::vector<paddle::Tensor> ProdForceSeAMaskGradOpCPUForward(
    const paddle::Tensor& grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& mask_tensor,
    const paddle::Tensor& nlist_tensor,
    int total_atom_num) {
  CHECK_INPUT(grad_tensor);
  CHECK_INPUT(net_deriv_tensor);
  CHECK_INPUT(in_deriv_tensor);
  CHECK_INPUT(mask_tensor);
  CHECK_INPUT(nlist_tensor);

  CHECK_INPUT_DIM(grad_tensor, 2);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(mask_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);

  PD_CHECK(total_atom_num >= 3,
           "Number of atoms should be larger than (or equal to) 3");

  int nframes = net_deriv_tensor.shape()[0];
  int nloc = total_atom_num;
  int ndescrpt = nloc > 0 ? net_deriv_tensor.shape()[1] / nloc : 0;
  int nnei = total_atom_num;

  PD_CHECK(nframes == grad_tensor.shape()[0], "Number of frames should match");
  PD_CHECK(nframes == in_deriv_tensor.shape()[0],
           "Number of frames should match");
  PD_CHECK(nframes == nlist_tensor.shape()[0],
           "Number of frames should match");
  PD_CHECK(nframes == mask_tensor.shape()[0],
           "Number of frames should match");

  PD_CHECK(nloc * 3 == grad_tensor.shape()[1], "Input grad shape is incorrect");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1],
           "Number of descriptors should match");

  // Create an output tensor
  std::vector<int64_t> grad_net_shape{nframes, nloc * ndescrpt};
  paddle::Tensor grad_net_tensor =
      paddle::empty(grad_net_shape, grad_tensor.dtype(), grad_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      grad_tensor.type(), "prod_force_se_a_mask_grad_cpu_forward_kernel", ([&] {
        ProdForceSeAMaskGradOpCPUKernel<data_t>(
            nframes, total_atom_num, grad_tensor.data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            mask_tensor.data<int>(), nlist_tensor.data<int>(),
            grad_net_tensor.data<data_t>());
      }));

  return {grad_net_tensor};
}

std::vector<paddle::Tensor> ProdForceSeAMaskGradForward(
    const paddle::Tensor& grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& mask_tensor,
    const paddle::Tensor& nlist_tensor,
    int total_atom_num) {
  return ProdForceSeAMaskGradOpCPUForward(grad_tensor, net_deriv_tensor,
                                          in_deriv_tensor, mask_tensor,
                                          nlist_tensor, total_atom_num);
}

std::vector<std::vector<int64_t>> ProdForceSeAMaskGradInferShape(
    std::vector<int64_t> grad_shape,
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> mask_shape,
    std::vector<int64_t> nlist_shape,
    int total_atom_num) {
  int nloc = total_atom_num;
  int ndescrpt = nloc > 0 ? grad_shape[1] / nloc : 0;

  PD_CHECK(grad_shape.size() == 2, "Dim of grad should be 2");
  PD_CHECK(net_deriv_shape.size() == 2, "Dim of net deriv should be 2");
  PD_CHECK(in_deriv_shape.size() == 2, "Dim of input deriv should be 2");
  PD_CHECK(mask_shape.size() == 2, "Dim of mask should be 2");
  PD_CHECK(nlist_shape.size() == 2, "Dim of nlist should be 2");

  PD_CHECK(grad_shape[0] > 0, "Number of frames should be greater than 0");
  PD_CHECK(grad_shape[0] == net_deriv_shape[0],
           "Number of frames should match");
  PD_CHECK(grad_shape[0] == in_deriv_shape[0], "Number of frames should match");
  PD_CHECK(grad_shape[0] == mask_shape[0], "Number of frames should match");
  PD_CHECK(grad_shape[0] == nlist_shape[0], "Number of frames should match");

  PD_CHECK(nloc * 3 == grad_shape[1], "Input grad shape is incorrect");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1],
           "Number of descriptors should match");

  std::vector<int64_t> grad_net_shape{nloc * ndescrpt};
  return {grad_net_shape};
}

std::vector<paddle::DataType> ProdForceSeAMaskGradInferDtype(
    paddle::DataType grad_dtype,
    paddle::DataType net_deriv_dtype,
    paddle::DataType in_deriv_dtype,
    paddle::DataType mask_dtype,
    paddle::DataType nlist_dtype) {
  PD_CHECK(grad_dtype == net_deriv_dtype && net_deriv_dtype == in_deriv_dtype,
           "Input data types should match");
  PD_CHECK(mask_dtype == nlist_dtype, "Input data types should match");

  return {grad_dtype};
}

PD_BUILD_OP(prod_force_se_a_mask_grad)
    .Inputs({"grad", "net_deriv", "in_deriv", "mask", "nlist"})
    .Attrs({"total_atom_num: int"})
    .Outputs({"grad_net"})
    .SetKernelFn(PD_KERNEL(ProdForceSeAMaskGradForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ProdForceSeAMaskGradInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ProdForceSeAMaskGradInferDtype));
