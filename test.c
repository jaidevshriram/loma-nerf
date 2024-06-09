#include <stdio.h>
#include <math.h>
        
typedef struct {
        float val;
        float dval;
} _dfloat;
float mlp_fit(float** layer_input, int layer_input_h, int layer_input_w, float** layer_output, float*** ws, float*** bs, float** target_image, int target_image_h, int target_image_w, int num_weights, int** weight_shapes, int** bias_shapes, int** intermediate_output_shapes, float*** intermediate_outputs);
void grad_mlp_fit(float** layer_input, float** _dlayer_input_2AJMqc, int layer_input_h, int* _dlayer_input_h_ow7I4q, int layer_input_w, int* _dlayer_input_w_WKOti9, float** layer_output, float** _dlayer_output_M8VcBy, float*** ws, float*** _dws_cF38jJ, float*** bs, float*** _dbs_fJvod8, float** target_image, float** _dtarget_image_ABnFdo, int target_image_h, int* _dtarget_image_h_Fwr8RL, int target_image_w, int* _dtarget_image_w_b80yTP, int num_weights, int* _dnum_weights_NgdO6w, int** weight_shapes, int** _dweight_shapes_ua2op4, int** bias_shapes, int** _dbias_shapes_PMi7tg, int** intermediate_output_shapes, int** _dintermediate_output_shapes_hTDzdf, float*** intermediate_outputs, float*** _dintermediate_outputs_ERnVPk, float _dreturn_omnPUy);
_dfloat make__dfloat(float val, float dval);
float mlp_fit(float** layer_input, int layer_input_h, int layer_input_w, float** layer_output, float*** ws, float*** bs, float** target_image, int target_image_h, int target_image_w, int num_weights, int** weight_shapes, int** bias_shapes, int** intermediate_output_shapes, float*** intermediate_outputs) {
        int i = (int)(0);
        int j = (int)(0);
        int k = (int)(0);
        int batch_num = (int)(0);
        int layer_counter = (int)(0);
        int num_layers = (int)(3);
        int i_mult = (int)(0);
        int j_mult = (int)(0);
        int k_mult = (int)(0);
        int i_relu = (int)(0);
        int j_relu = (int)(0);
        int i_mse = (int)(0);
        int j_mse = (int)(0);
        while ((layer_counter) < (num_layers)) {
                if ((layer_counter) == ((int)(0))) {
                        i_mult = (int)(0);
                        j_mult = (int)(0);
                        k_mult = (int)(0);
                        while ((i_mult) < (((weight_shapes)[layer_counter])[(int)(0)])) {
                                j_mult = (int)(0);
                                while ((j_mult) < (((weight_shapes)[layer_counter])[(int)(1)])) {
                                        k_mult = (int)(0);
                                        while ((k_mult) < (layer_input_w)) {
                                                (((intermediate_outputs)[layer_counter])[i_mult])[j_mult] = ((((intermediate_outputs)[layer_counter])[i_mult])[j_mult]) + (((((ws)[layer_counter])[i_mult])[j_mult]) * (((layer_input)[i_mult])[k_mult]));
                                                k_mult = (k_mult) + ((int)(1));
                                        }
                                        j_mult = (j_mult) + ((int)(1));
                                }
                                i_mult = (i_mult) + ((int)(1));
                        }
                } else {
                        i_mult = (int)(0);
                        j_mult = (int)(0);
                        k_mult = (int)(0);
                        while ((i_mult) < (((weight_shapes)[layer_counter])[(int)(0)])) {
                                j_mult = (int)(0);
                                while ((j_mult) < (((weight_shapes)[layer_counter])[(int)(1)])) {
                                        k_mult = (int)(0);
                                        while ((k_mult) < (((intermediate_output_shapes)[(layer_counter) - ((int)(1))])[(int)(1)])) {
                                                (((intermediate_outputs)[layer_counter])[i_mult])[j_mult] = ((((intermediate_outputs)[layer_counter])[i_mult])[j_mult]) + (((((ws)[layer_counter])[i_mult])[j_mult]) * ((((intermediate_outputs)[(layer_counter) - ((int)(1))])[i_mult])[k_mult]));
                                                k_mult = (k_mult) + ((int)(1));
                                        }
                                        j_mult = (j_mult) + ((int)(1));
                                }
                                i_mult = (i_mult) + ((int)(1));
                        }
                }
                i_relu = (int)(0);
                j_relu = (int)(0);
                while ((i_relu) < (((intermediate_output_shapes)[layer_counter])[(int)(0)])) {
                        j_relu = (int)(0);
                        while ((j_relu) < (((intermediate_output_shapes)[layer_counter])[(int)(1)])) {
                                if (((((intermediate_outputs)[layer_counter])[i_relu])[j_relu]) > ((float)((int)(0)))) {
                                        (((intermediate_outputs)[layer_counter])[i_relu])[j_relu] = (((intermediate_outputs)[layer_counter])[i_relu])[j_relu];
                                } else {
                                        (((intermediate_outputs)[layer_counter])[i_relu])[j_relu] = (float)((int)(0));
                                }
                                j_relu = (j_relu) + ((int)(1));
                        }
                        i_relu = (i_relu) + ((int)(1));
                }
                layer_counter = (layer_counter) + ((int)(1));
        }
        float loss = (float)((int)(0));
        i_mse = (int)(0);
        j_mse = (int)(0);
        while ((i_mse) < (target_image_h)) {
                j_mse = (int)(0);
                while ((j_mse) < (target_image_w)) {
                        loss = (loss) + (((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])) * ((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])));
                        j_mse = (j_mse) + ((int)(1));
                }
                i_mse = (i_mse) + ((int)(1));
        }
        return loss;
}

void grad_mlp_fit(float** layer_input, float** _dlayer_input_2AJMqc, int layer_input_h, int* _dlayer_input_h_ow7I4q, int layer_input_w, int* _dlayer_input_w_WKOti9, float** layer_output, float** _dlayer_output_M8VcBy, float*** ws, float*** _dws_cF38jJ, float*** bs, float*** _dbs_fJvod8, float** target_image, float** _dtarget_image_ABnFdo, int target_image_h, int* _dtarget_image_h_Fwr8RL, int target_image_w, int* _dtarget_image_w_b80yTP, int num_weights, int* _dnum_weights_NgdO6w, int** weight_shapes, int** _dweight_shapes_ua2op4, int** bias_shapes, int** _dbias_shapes_PMi7tg, int** intermediate_output_shapes, int** _dintermediate_output_shapes_hTDzdf, float*** intermediate_outputs, float*** _dintermediate_outputs_ERnVPk, float _dreturn_omnPUy) {
        int _t_int_q1e0nO[266212503];
        for (int _i = 0; _i < 266212503;_i++) {
                _t_int_q1e0nO[_i] = 0;
        }
        int _stack_ptr_int_q1e0nO = (int)(0);
        float _t_float_kFq87T[257502500];
        for (int _i = 0; _i < 257502500;_i++) {
                _t_float_kFq87T[_i] = 0;
        }
        int _stack_ptr_float_kFq87T = (int)(0);
        int _loop_var_0_Erwkue;
        _loop_var_0_Erwkue = 0;
        int _loop_var_1_8hEV3t;
        _loop_var_1_8hEV3t = 0;
        int _loop_var_2_gYNR3G;
        _loop_var_2_gYNR3G = 0;
        int _loop_var_3_mYR4sM;
        _loop_var_3_mYR4sM = 0;
        int _loop_var_3_mYR4sM_stack[2500000];
        for (int _i = 0; _i < 2500000;_i++) {
                _loop_var_3_mYR4sM_stack[_i] = 0;
        }
        int _loop_var_3_mYR4sM_stack_ptr;
        _loop_var_3_mYR4sM_stack_ptr = 0;
        int _loop_var_2_gYNR3G_stack[50000];
        for (int _i = 0; _i < 50000;_i++) {
                _loop_var_2_gYNR3G_stack[_i] = 0;
        }
        int _loop_var_2_gYNR3G_stack_ptr;
        _loop_var_2_gYNR3G_stack_ptr = 0;
        int _loop_var_1_8hEV3t_stack[1000];
        for (int _i = 0; _i < 1000;_i++) {
                _loop_var_1_8hEV3t_stack[_i] = 0;
        }
        int _loop_var_1_8hEV3t_stack_ptr;
        _loop_var_1_8hEV3t_stack_ptr = 0;
        int _loop_var_4_11aY5O;
        _loop_var_4_11aY5O = 0;
        int _loop_var_5_XyB7sl;
        _loop_var_5_XyB7sl = 0;
        int _loop_var_6_QVvYk8;
        _loop_var_6_QVvYk8 = 0;
        int _loop_var_6_QVvYk8_stack[2500000];
        for (int _i = 0; _i < 2500000;_i++) {
                _loop_var_6_QVvYk8_stack[_i] = 0;
        }
        int _loop_var_6_QVvYk8_stack_ptr;
        _loop_var_6_QVvYk8_stack_ptr = 0;
        int _loop_var_5_XyB7sl_stack[50000];
        for (int _i = 0; _i < 50000;_i++) {
                _loop_var_5_XyB7sl_stack[_i] = 0;
        }
        int _loop_var_5_XyB7sl_stack_ptr;
        _loop_var_5_XyB7sl_stack_ptr = 0;
        int _loop_var_4_11aY5O_stack[1000];
        for (int _i = 0; _i < 1000;_i++) {
                _loop_var_4_11aY5O_stack[_i] = 0;
        }
        int _loop_var_4_11aY5O_stack_ptr;
        _loop_var_4_11aY5O_stack_ptr = 0;
        int _loop_var_7_5GbOqV;
        _loop_var_7_5GbOqV = 0;
        int _loop_var_8_qyKZrD;
        _loop_var_8_qyKZrD = 0;
        int _loop_var_8_qyKZrD_stack[500000];
        for (int _i = 0; _i < 500000;_i++) {
                _loop_var_8_qyKZrD_stack[_i] = 0;
        }
        int _loop_var_8_qyKZrD_stack_ptr;
        _loop_var_8_qyKZrD_stack_ptr = 0;
        int _loop_var_7_5GbOqV_stack[1000];
        for (int _i = 0; _i < 1000;_i++) {
                _loop_var_7_5GbOqV_stack[_i] = 0;
        }
        int _loop_var_7_5GbOqV_stack_ptr;
        _loop_var_7_5GbOqV_stack_ptr = 0;
        int _loop_var_9_okrJtH;
        _loop_var_9_okrJtH = 0;
        int _loop_var_10_N7ryks;
        _loop_var_10_N7ryks = 0;
        int _loop_var_10_N7ryks_stack[500];
        for (int _i = 0; _i < 500;_i++) {
                _loop_var_10_N7ryks_stack[_i] = 0;
        }
        int _loop_var_10_N7ryks_stack_ptr;
        _loop_var_10_N7ryks_stack_ptr = 0;
        int _call_t_0_fplr1H;
        _call_t_0_fplr1H = 0;
        int _call_t_1_rZKz8S;
        _call_t_1_rZKz8S = 0;
        float _call_t_3_gnzU18;
        _call_t_3_gnzU18 = 0;
        float _d_call_t_3_gnzU18_BHTUHZ;
        _d_call_t_3_gnzU18_BHTUHZ = 0;
        int _call_t_4_AMRm6B;
        _call_t_4_AMRm6B = 0;
        int i = (int)(0);
        int j = (int)(0);
        int k = (int)(0);
        int batch_num = (int)(0);
        int layer_counter = (int)(0);
        int num_layers = (int)(3);
        int i_mult = (int)(0);
        int j_mult = (int)(0);
        int k_mult = (int)(0);
        int i_relu = (int)(0);
        int j_relu = (int)(0);
        int i_mse = (int)(0);
        int j_mse = (int)(0);
        _loop_var_0_Erwkue = (int)(0);
        while ((layer_counter) < (num_layers)) {
                if ((layer_counter) == ((int)(0))) {
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_mult;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        i_mult = (int)(0);
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mult;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        j_mult = (int)(0);
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = k_mult;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        k_mult = (int)(0);
                        _loop_var_1_8hEV3t = (int)(0);
                        while ((i_mult) < (((weight_shapes)[layer_counter])[(int)(0)])) {
                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mult;
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                j_mult = (int)(0);
                                _loop_var_2_gYNR3G = (int)(0);
                                while ((j_mult) < (((weight_shapes)[layer_counter])[(int)(1)])) {
                                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = k_mult;
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                        k_mult = (int)(0);
                                        _loop_var_3_mYR4sM = (int)(0);
                                        while ((k_mult) < (layer_input_w)) {
                                                (_t_float_kFq87T)[_stack_ptr_float_kFq87T] = (((intermediate_outputs)[layer_counter])[i_mult])[j_mult];
                                                _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) + ((int)(1));
                                                (((intermediate_outputs)[layer_counter])[i_mult])[j_mult] = ((((intermediate_outputs)[layer_counter])[i_mult])[j_mult]) + (((((ws)[layer_counter])[i_mult])[j_mult]) * (((layer_input)[i_mult])[k_mult]));
                                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = k_mult;
                                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                                k_mult = (k_mult) + ((int)(1));
                                                _loop_var_3_mYR4sM = (_loop_var_3_mYR4sM) + ((int)(1));
                                        }
                                        (_loop_var_3_mYR4sM_stack)[_loop_var_3_mYR4sM_stack_ptr] = _loop_var_3_mYR4sM;
                                        _loop_var_3_mYR4sM_stack_ptr = (_loop_var_3_mYR4sM_stack_ptr) + ((int)(1));
                                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mult;
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                        j_mult = (j_mult) + ((int)(1));
                                        _loop_var_2_gYNR3G = (_loop_var_2_gYNR3G) + ((int)(1));
                                }
                                (_loop_var_2_gYNR3G_stack)[_loop_var_2_gYNR3G_stack_ptr] = _loop_var_2_gYNR3G;
                                _loop_var_2_gYNR3G_stack_ptr = (_loop_var_2_gYNR3G_stack_ptr) + ((int)(1));
                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_mult;
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                i_mult = (i_mult) + ((int)(1));
                                _loop_var_1_8hEV3t = (_loop_var_1_8hEV3t) + ((int)(1));
                        }
                        (_loop_var_1_8hEV3t_stack)[_loop_var_1_8hEV3t_stack_ptr] = _loop_var_1_8hEV3t;
                        _loop_var_1_8hEV3t_stack_ptr = (_loop_var_1_8hEV3t_stack_ptr) + ((int)(1));
                } else {
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_mult;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        i_mult = (int)(0);
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mult;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        j_mult = (int)(0);
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = k_mult;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        k_mult = (int)(0);
                        _loop_var_4_11aY5O = (int)(0);
                        while ((i_mult) < (((weight_shapes)[layer_counter])[(int)(0)])) {
                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mult;
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                j_mult = (int)(0);
                                _loop_var_5_XyB7sl = (int)(0);
                                while ((j_mult) < (((weight_shapes)[layer_counter])[(int)(1)])) {
                                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = k_mult;
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                        k_mult = (int)(0);
                                        _loop_var_6_QVvYk8 = (int)(0);
                                        while ((k_mult) < (((intermediate_output_shapes)[(layer_counter) - ((int)(1))])[(int)(1)])) {
                                                (_t_float_kFq87T)[_stack_ptr_float_kFq87T] = (((intermediate_outputs)[layer_counter])[i_mult])[j_mult];
                                                _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) + ((int)(1));
                                                (((intermediate_outputs)[layer_counter])[i_mult])[j_mult] = ((((intermediate_outputs)[layer_counter])[i_mult])[j_mult]) + (((((ws)[layer_counter])[i_mult])[j_mult]) * ((((intermediate_outputs)[(layer_counter) - ((int)(1))])[i_mult])[k_mult]));
                                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = k_mult;
                                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                                k_mult = (k_mult) + ((int)(1));
                                                _loop_var_6_QVvYk8 = (_loop_var_6_QVvYk8) + ((int)(1));
                                        }
                                        (_loop_var_6_QVvYk8_stack)[_loop_var_6_QVvYk8_stack_ptr] = _loop_var_6_QVvYk8;
                                        _loop_var_6_QVvYk8_stack_ptr = (_loop_var_6_QVvYk8_stack_ptr) + ((int)(1));
                                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mult;
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                        j_mult = (j_mult) + ((int)(1));
                                        _loop_var_5_XyB7sl = (_loop_var_5_XyB7sl) + ((int)(1));
                                }
                                (_loop_var_5_XyB7sl_stack)[_loop_var_5_XyB7sl_stack_ptr] = _loop_var_5_XyB7sl;
                                _loop_var_5_XyB7sl_stack_ptr = (_loop_var_5_XyB7sl_stack_ptr) + ((int)(1));
                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_mult;
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                i_mult = (i_mult) + ((int)(1));
                                _loop_var_4_11aY5O = (_loop_var_4_11aY5O) + ((int)(1));
                        }
                        (_loop_var_4_11aY5O_stack)[_loop_var_4_11aY5O_stack_ptr] = _loop_var_4_11aY5O;
                        _loop_var_4_11aY5O_stack_ptr = (_loop_var_4_11aY5O_stack_ptr) + ((int)(1));
                }
                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_relu;
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                i_relu = (int)(0);
                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_relu;
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                j_relu = (int)(0);
                _loop_var_7_5GbOqV = (int)(0);
                while ((i_relu) < (((intermediate_output_shapes)[layer_counter])[(int)(0)])) {
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_relu;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        j_relu = (int)(0);
                        _loop_var_8_qyKZrD = (int)(0);
                        while ((j_relu) < (((intermediate_output_shapes)[layer_counter])[(int)(1)])) {
                                if (((((intermediate_outputs)[layer_counter])[i_relu])[j_relu]) > ((float)(_call_t_0_fplr1H))) {
                                        (_t_float_kFq87T)[_stack_ptr_float_kFq87T] = (((intermediate_outputs)[layer_counter])[i_relu])[j_relu];
                                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) + ((int)(1));
                                        (((intermediate_outputs)[layer_counter])[i_relu])[j_relu] = (((intermediate_outputs)[layer_counter])[i_relu])[j_relu];
                                } else {
                                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = _call_t_1_rZKz8S;
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                        _call_t_1_rZKz8S = (int)(0);
                                        (_t_float_kFq87T)[_stack_ptr_float_kFq87T] = _call_t_3_gnzU18;
                                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) + ((int)(1));
                                        _call_t_3_gnzU18 = (float)(_call_t_1_rZKz8S);
                                        (_t_float_kFq87T)[_stack_ptr_float_kFq87T] = (((intermediate_outputs)[layer_counter])[i_relu])[j_relu];
                                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) + ((int)(1));
                                        (((intermediate_outputs)[layer_counter])[i_relu])[j_relu] = _call_t_3_gnzU18;
                                }
                                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_relu;
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                                j_relu = (j_relu) + ((int)(1));
                                _loop_var_8_qyKZrD = (_loop_var_8_qyKZrD) + ((int)(1));
                        }
                        (_loop_var_8_qyKZrD_stack)[_loop_var_8_qyKZrD_stack_ptr] = _loop_var_8_qyKZrD;
                        _loop_var_8_qyKZrD_stack_ptr = (_loop_var_8_qyKZrD_stack_ptr) + ((int)(1));
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_relu;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        i_relu = (i_relu) + ((int)(1));
                        _loop_var_7_5GbOqV = (_loop_var_7_5GbOqV) + ((int)(1));
                }
                (_loop_var_7_5GbOqV_stack)[_loop_var_7_5GbOqV_stack_ptr] = _loop_var_7_5GbOqV;
                _loop_var_7_5GbOqV_stack_ptr = (_loop_var_7_5GbOqV_stack_ptr) + ((int)(1));
                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = layer_counter;
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                layer_counter = (layer_counter) + ((int)(1));
                _loop_var_0_Erwkue = (_loop_var_0_Erwkue) + ((int)(1));
        }
        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = _call_t_4_AMRm6B;
        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
        _call_t_4_AMRm6B = (int)(0);
        float loss = (float)(_call_t_4_AMRm6B);
        float _dloss_grsUu6;
        _dloss_grsUu6 = 0;
        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_mse;
        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
        i_mse = (int)(0);
        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mse;
        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
        j_mse = (int)(0);
        _loop_var_9_okrJtH = (int)(0);
        while ((i_mse) < (target_image_h)) {
                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mse;
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                j_mse = (int)(0);
                _loop_var_10_N7ryks = (int)(0);
                while ((j_mse) < (target_image_w)) {
                        (_t_float_kFq87T)[_stack_ptr_float_kFq87T] = loss;
                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) + ((int)(1));
                        loss = (loss) + (((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])) * ((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])));
                        (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = j_mse;
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                        j_mse = (j_mse) + ((int)(1));
                        _loop_var_10_N7ryks = (_loop_var_10_N7ryks) + ((int)(1));
                }
                (_loop_var_10_N7ryks_stack)[_loop_var_10_N7ryks_stack_ptr] = _loop_var_10_N7ryks;
                _loop_var_10_N7ryks_stack_ptr = (_loop_var_10_N7ryks_stack_ptr) + ((int)(1));
                (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO] = i_mse;
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) + ((int)(1));
                i_mse = (i_mse) + ((int)(1));
                _loop_var_9_okrJtH = (_loop_var_9_okrJtH) + ((int)(1));
        }
        float _adj_0;
        _adj_0 = 0;
        float _adj_1;
        _adj_1 = 0;
        float _adj_2;
        _adj_2 = 0;
        float _adj_3;
        _adj_3 = 0;
        float _adj_4;
        _adj_4 = 0;
        float _adj_5;
        _adj_5 = 0;
        float _adj_6;
        _adj_6 = 0;
        float _adj_7;
        _adj_7 = 0;
        float _adj_8;
        _adj_8 = 0;
        float _adj_9;
        _adj_9 = 0;
        float _adj_10;
        _adj_10 = 0;
        float _adj_11;
        _adj_11 = 0;
        float _adj_12;
        _adj_12 = 0;
        _dloss_grsUu6 += _dreturn_omnPUy;
        while ((_loop_var_9_okrJtH) > ((int)(0))) {
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                i_mse = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                _loop_var_10_N7ryks_stack_ptr = (_loop_var_10_N7ryks_stack_ptr) - ((int)(1));
                _loop_var_10_N7ryks = (_loop_var_10_N7ryks_stack)[_loop_var_10_N7ryks_stack_ptr];
                while ((_loop_var_10_N7ryks) > ((int)(0))) {
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        j_mse = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) - ((int)(1));
                        loss = (_t_float_kFq87T)[_stack_ptr_float_kFq87T];
                        _adj_0 = _dloss_grsUu6;
                        _adj_1 = ((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])) * (_dloss_grsUu6);
                        _adj_2 = ((float)(0.0)) - (((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])) * (_dloss_grsUu6));
                        _adj_3 = ((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])) * (_dloss_grsUu6);
                        _adj_4 = ((float)(0.0)) - (((((layer_input)[i_mse])[j_mse]) - (((target_image)[i_mse])[j_mse])) * (_dloss_grsUu6));
                        _dloss_grsUu6 = (float)(0.0);
                        _dloss_grsUu6 += _adj_0;
                        ((_dlayer_input_2AJMqc)[i_mse])[j_mse] += _adj_1;
                        ((_dtarget_image_ABnFdo)[i_mse])[j_mse] += _adj_2;
                        ((_dlayer_input_2AJMqc)[i_mse])[j_mse] += _adj_3;
                        ((_dtarget_image_ABnFdo)[i_mse])[j_mse] += _adj_4;
                        _loop_var_10_N7ryks = (_loop_var_10_N7ryks) - ((int)(1));
                }
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                j_mse = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                _loop_var_9_okrJtH = (_loop_var_9_okrJtH) - ((int)(1));
        }
        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
        j_mse = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
        i_mse = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
        _call_t_4_AMRm6B = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
        while ((_loop_var_0_Erwkue) > ((int)(0))) {
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                layer_counter = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                _loop_var_7_5GbOqV_stack_ptr = (_loop_var_7_5GbOqV_stack_ptr) - ((int)(1));
                _loop_var_7_5GbOqV = (_loop_var_7_5GbOqV_stack)[_loop_var_7_5GbOqV_stack_ptr];
                while ((_loop_var_7_5GbOqV) > ((int)(0))) {
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        i_relu = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _loop_var_8_qyKZrD_stack_ptr = (_loop_var_8_qyKZrD_stack_ptr) - ((int)(1));
                        _loop_var_8_qyKZrD = (_loop_var_8_qyKZrD_stack)[_loop_var_8_qyKZrD_stack_ptr];
                        while ((_loop_var_8_qyKZrD) > ((int)(0))) {
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                j_relu = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                if (((((intermediate_outputs)[layer_counter])[i_relu])[j_relu]) > ((float)(_call_t_0_fplr1H))) {
                                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) - ((int)(1));
                                        (((intermediate_outputs)[layer_counter])[i_relu])[j_relu] = (_t_float_kFq87T)[_stack_ptr_float_kFq87T];
                                        _adj_6 = (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_relu])[j_relu];
                                        (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_relu])[j_relu] = (float)(0.0);
                                        (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_relu])[j_relu] += _adj_6;
                                } else {
                                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) - ((int)(1));
                                        (((intermediate_outputs)[layer_counter])[i_relu])[j_relu] = (_t_float_kFq87T)[_stack_ptr_float_kFq87T];
                                        _adj_5 = (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_relu])[j_relu];
                                        (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_relu])[j_relu] = (float)(0.0);
                                        _d_call_t_3_gnzU18_BHTUHZ += _adj_5;
                                        _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) - ((int)(1));
                                        _call_t_3_gnzU18 = (_t_float_kFq87T)[_stack_ptr_float_kFq87T];
                                        _d_call_t_3_gnzU18_BHTUHZ = (float)(0.0);
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                        _call_t_1_rZKz8S = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                }
                                _loop_var_8_qyKZrD = (_loop_var_8_qyKZrD) - ((int)(1));
                        }
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        j_relu = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _loop_var_7_5GbOqV = (_loop_var_7_5GbOqV) - ((int)(1));
                }
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                j_relu = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                i_relu = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                if ((layer_counter) == ((int)(0))) {
                        _loop_var_1_8hEV3t_stack_ptr = (_loop_var_1_8hEV3t_stack_ptr) - ((int)(1));
                        _loop_var_1_8hEV3t = (_loop_var_1_8hEV3t_stack)[_loop_var_1_8hEV3t_stack_ptr];
                        while ((_loop_var_1_8hEV3t) > ((int)(0))) {
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                i_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                _loop_var_2_gYNR3G_stack_ptr = (_loop_var_2_gYNR3G_stack_ptr) - ((int)(1));
                                _loop_var_2_gYNR3G = (_loop_var_2_gYNR3G_stack)[_loop_var_2_gYNR3G_stack_ptr];
                                while ((_loop_var_2_gYNR3G) > ((int)(0))) {
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                        j_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                        _loop_var_3_mYR4sM_stack_ptr = (_loop_var_3_mYR4sM_stack_ptr) - ((int)(1));
                                        _loop_var_3_mYR4sM = (_loop_var_3_mYR4sM_stack)[_loop_var_3_mYR4sM_stack_ptr];
                                        while ((_loop_var_3_mYR4sM) > ((int)(0))) {
                                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                                k_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                                _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) - ((int)(1));
                                                (((intermediate_outputs)[layer_counter])[i_mult])[j_mult] = (_t_float_kFq87T)[_stack_ptr_float_kFq87T];
                                                _adj_10 = (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult];
                                                _adj_11 = (((layer_input)[i_mult])[k_mult]) * ((((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult]);
                                                _adj_12 = ((((ws)[layer_counter])[i_mult])[j_mult]) * ((((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult]);
                                                (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult] = (float)(0.0);
                                                (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult] += _adj_10;
                                                (((_dws_cF38jJ)[layer_counter])[i_mult])[j_mult] += _adj_11;
                                                ((_dlayer_input_2AJMqc)[i_mult])[k_mult] += _adj_12;
                                                _loop_var_3_mYR4sM = (_loop_var_3_mYR4sM) - ((int)(1));
                                        }
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                        k_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                        _loop_var_2_gYNR3G = (_loop_var_2_gYNR3G) - ((int)(1));
                                }
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                j_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                _loop_var_1_8hEV3t = (_loop_var_1_8hEV3t) - ((int)(1));
                        }
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        k_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        j_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        i_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                } else {
                        _loop_var_4_11aY5O_stack_ptr = (_loop_var_4_11aY5O_stack_ptr) - ((int)(1));
                        _loop_var_4_11aY5O = (_loop_var_4_11aY5O_stack)[_loop_var_4_11aY5O_stack_ptr];
                        while ((_loop_var_4_11aY5O) > ((int)(0))) {
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                i_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                _loop_var_5_XyB7sl_stack_ptr = (_loop_var_5_XyB7sl_stack_ptr) - ((int)(1));
                                _loop_var_5_XyB7sl = (_loop_var_5_XyB7sl_stack)[_loop_var_5_XyB7sl_stack_ptr];
                                while ((_loop_var_5_XyB7sl) > ((int)(0))) {
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                        j_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                        _loop_var_6_QVvYk8_stack_ptr = (_loop_var_6_QVvYk8_stack_ptr) - ((int)(1));
                                        _loop_var_6_QVvYk8 = (_loop_var_6_QVvYk8_stack)[_loop_var_6_QVvYk8_stack_ptr];
                                        while ((_loop_var_6_QVvYk8) > ((int)(0))) {
                                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                                k_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                                _stack_ptr_float_kFq87T = (_stack_ptr_float_kFq87T) - ((int)(1));
                                                (((intermediate_outputs)[layer_counter])[i_mult])[j_mult] = (_t_float_kFq87T)[_stack_ptr_float_kFq87T];
                                                _adj_7 = (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult];
                                                _adj_8 = ((((intermediate_outputs)[(layer_counter) - ((int)(1))])[i_mult])[k_mult]) * ((((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult]);
                                                _adj_9 = ((((ws)[layer_counter])[i_mult])[j_mult]) * ((((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult]);
                                                (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult] = (float)(0.0);
                                                (((_dintermediate_outputs_ERnVPk)[layer_counter])[i_mult])[j_mult] += _adj_7;
                                                (((_dws_cF38jJ)[layer_counter])[i_mult])[j_mult] += _adj_8;
                                                (((_dintermediate_outputs_ERnVPk)[(layer_counter) - ((int)(1))])[i_mult])[k_mult] += _adj_9;
                                                _loop_var_6_QVvYk8 = (_loop_var_6_QVvYk8) - ((int)(1));
                                        }
                                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                        k_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                        _loop_var_5_XyB7sl = (_loop_var_5_XyB7sl) - ((int)(1));
                                }
                                _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                                j_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                                _loop_var_4_11aY5O = (_loop_var_4_11aY5O) - ((int)(1));
                        }
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        k_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        j_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                        _stack_ptr_int_q1e0nO = (_stack_ptr_int_q1e0nO) - ((int)(1));
                        i_mult = (_t_int_q1e0nO)[_stack_ptr_int_q1e0nO];
                }
                _loop_var_0_Erwkue = (_loop_var_0_Erwkue) - ((int)(1));
        }
}
_dfloat make__dfloat(float val, float dval) {
        _dfloat ret;
        ret.val = 0;
        ret.dval = 0;
        (ret).val = val;
        (ret).dval = dval;
        return ret;
}

void main() {

    int _t_int_q1e0nO[26621];
    for (int _i = 0; _i < 2603;_i++) {
            _t_int_q1e0nO[_i] = 0;
    }
    int _stack_ptr_int_q1e0nO = (int)(0);
    float _t_float_kFq87T[257];
    for (int _i = 0; _i < 25;_i++) {
            _t_float_kFq87T[_i] = 0;
    }

    // Print hello
    printf("Hello, world!\n");
}