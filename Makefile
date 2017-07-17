TARGETS          = vae dcgan ebgan began ali
COND_TARGETS     = cvae cvaegan cali
IM2IM_TARGETS    = unit
OPTIONS          = --dataset=mnist --zdims=128 --epoch=1 --testmode
OPTIONS_IM2IM    = --first-data=mnist --second-data=svhn --zdims=128 --epoch=1 --testmode

# Test method for basic models
define runtest
$(1):
	python train.py --model=$(1) $(OPTIONS)
endef

# Test method for conditional models
define runtest_conditional
$(1):
	python train_conditional.py --model=$(1) $(OPTIONS)
endef

# Test method for image-to-image models
define runtest_im2im
$(1):
	python train_im2im.py --model=$(1)
endef

# Check all types of models
check: check_basic check_conditional

# Test basic models
check_basic: $(TARGETS)
$(foreach model,$(TARGETS), $(eval $(call runtest,$(model))))

# Test conditional models
check_conditional: $(COND_TARGETS)
$(foreach model,$(COND_TARGETS), $(eval $(call runtest_conditional,$(model))))

# Test image-to-image models
check_im2im: $(IM2IM TARGETS)
$(foreach model, $(IM2IM_TARGETS), $(eval $(call runtest_im2im,$(model))))
