TARGETS      = vae dcgan ebgan began ali

define runtest
$(1):
	python train.py --model=$(1) --dataset=mnist --zdims=128 --epoch=1 --testmode
endef

check: $(TARGETS)

$(foreach model,$(TARGETS), $(eval $(call runtest,$(model))))
