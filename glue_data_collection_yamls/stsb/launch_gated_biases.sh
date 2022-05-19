mcli sweep -f gated_gelu_w0_false_w1_false.yaml -y
sleep 10
mcli sweep -f gated_gelu_w0_false_w1_true.yaml -y
sleep 10
mcli sweep -f gated_gelu_w0_true_w1_false.yaml -y
sleep 10
mcli sweep -f gated_relu_w0_false_w1_false.yaml -y
sleep 10
mcli sweep -f gated_relu_w0_false_w1_true.yaml -y
sleep 10
mcli sweep -f gated_relu_w0_true_w1_false.yaml -y
