mcli sweep -f baseline.yaml -y --priority low
sleep 10
mcli sweep -f deepspeed.yaml -y --priority low
sleep 10
mcli sweep -f fln_deepspeed.yaml -y --priority low
sleep 10
mcli sweep -f fln.yaml -y --priority low
sleep 10
mcli sweep -f gated_gelu.yaml -y --priority low
sleep 10
mcli sweep -f gated_srelu.yaml -y --priority low
sleep 10
mcli sweep -f gated_relu.yaml -y --priority low
sleep 10
mcli sweep -f gated_swish.yaml -y --priority low
