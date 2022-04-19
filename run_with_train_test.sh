python run.py \
	--mc_experiment_number 11 \
	--mc_pretrained_model_name_or_path ./mc_ckpt/11/checkpoint-final \
	--do_mc_test \
	--mc_ckpt_dir ./mc_ckpt/ \
	--mc_pred_dir ./mc_pred/ \

python run.py \
	--mc_experiment_number 11 \
	--qa_experiment_number 8 \
	--do_qa_test \
	--qa_pretrained_model_name_or_path hfl/chinese-macbert-large \
	--qa_resume \
	--mc_pred_dir ./mc_pred/ \
	--qa_ckpt_dir ./qa_ckpt/ \
	--qa_pred_dir ./qa_pred/ \
