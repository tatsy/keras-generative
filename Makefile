TARGETS      = vae dcgan ebgan began ali
COND_TARGETS = cvae

define runtest
$(1):
	python train.py --model=$(1) --dataset=mnist --zdims=128 --epoch=1 --testmode
endef

define runtest_conditional
$(1):
	python train_conditional.py --model=$(1) --dataset=mnist --zdims=128 --epoch=1 --testmode
endef

check: check_basic check_conditional

check_basic: $(TARGETS)

$(foreach model,$(TARGETS), $(eval $(call runtest,$(model))))

check_conditional: $(COND_TARGETS)

$(foreach model,$(COND_TARGETS), $(eval $(call runtest_conditional,$(model))))
