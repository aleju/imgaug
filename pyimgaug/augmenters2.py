from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy
import re
import pyimgaug as ia
from parameters import StochasticParameter, Deterministic, Binomial, DiscreteUniform, Normal, Uniform

class Augmenter(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None, deterministic=False, random_state=None):
        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

        self.deterministic = deterministic

        if random_state is None:
            if self.deterministic:
                self.random_state = ia.new_random_state()
            else:
                self.random_state = ia.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.deactivated = False

    def transform(self, images):
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if isinstance(images, (list, tuple)):
            assert all([len(image.shape) == 3 for image in images])
            images_tf = list(images)
        elif ia.is_np_array(images):
            assert len(images.shape) == 4, "Expected 4d array of form (N, height, width, channels), got shape %s" % (str(images.shape),)
            assert images.dtype == np.uint8, "Expected dtype uint8 (with value range 0 to 255), got dtype %s." % (str(images.dtype),)
            images_tf = images
        else:
            raise Exception("Expected list/tuple of numpy arrays or one numpy array, got %s." % (type(images),))

        if self.deactivated:
            if isinstance(images_tf, list):
                images_result = [np.copy(image) for image in images]
            else:
                images_result = np.copy(images_tf)
        else:
            if isinstance(images_tf, list):
                images_result = []
                for image in images_tf:
                    images_result.append(self._transform(np.copy(np.array([image])), random_state=ia.copy_random_state(self.random_state)))
                    self.random_state.randint(0, 10**6)
            else:
                images_result = self._transform(np.copy(images_tf), random_state=ia.copy_random_state(self.random_state))
                self.random_state.randint(0, 10**6)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        return images_result

    @abstractmethod
    def _transform(self, images, random_state):
        raise NotImplemented()

    def transform_augjob(self, augjob):
        has_deactivator = (augjob.deactivator is not None)
        if (has_deactivator and augjob.deactivator(self, augjob)) or (not has_deactivator and self.deactivated):
            augjob.add_history(self, augjob.images, augjob.images, {})
            return augjob
        else:
            before = augjob.images

            if augjob.preprocessor is not None:
                augjob = augjob.preprocessor(self, augjob)

            augjob_transformed, changes = self._transform_augjob(augjob)

            if augjob.postprocessor is not None:
                augjob = augjob.postprocessor(self, augjob, augjob_transformed, changes)
            augjob.add_history(self, augjob.images, augjob_transformed.images, changes)

            return augjob_transformed

    def to_deterministic(self, n=None):
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in xrange(n)]

    def _to_deterministic(self):
        aug = copy.copy(self)
        aug.random_state = ia.new_random_state()
        aug.deterministic = True
        return aug

    @abstractmethod
    def get_parameters(self):
        raise NotImplemented()

    def get_children_lists(self):
        return []

    def find_augmenters(self, func, flat=True):
        return self._find(func, parents=[], flat=flat)

    def _find_augmenters(self, func, parents, flat):
        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                if flat:
                    result.extend(aug._find_augmenters(func, parents=subparents, flat=flat))
                else:
                    result.append(aug._find_augmenters(func, parents=subparents, flat=flat))
        return result

    def find_augmenters_by_name(self, name, regex=False, flat=True):
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        if regex:
            def comparer(aug, parents):
                for pattern in names:
                    if re.match(pattern, aug.name):
                        return True
                return False

            return self.find_augmenters(comparer, flat=flat)
        else:
            return self.find_augmenters(lambda aug, parents: aug.name in names, flat=flat)

    def remove_augmenters(self, func, copy=True, noop_if_topmost=True):
        if func(self, []):
            if not copy:
                raise Exception("Inplace removal of topmost augmenter requested, which is currently not possible.")

            if noop_if_topmost:
                return Noop()
            else:
                return None
        else:
            aug = self if not copy else self.deepcopy()
            aug._remove_augmenters_inplace(func, parents=[])
            return aug

    def _remove_augmenters_inplace(self, func, parents):
        subparents = parents + [self]
        for lst in self.get_children_lists():
            to_remove = []
            for i, aug in enumerate(lst):
                if func(aug, subparents):
                    to_remove.append((i, aug))

            for count_removed, (i, aug) in enumerate(to_remove):
                self._remove_augmenters_inplace_from_list(lst, aug, i, i - count_removed)

            for aug in lst:
                aug._remove_augmenters_inplace(func, subparents)

    def _remove_augmenters_inplace_from_list(self, lst, aug, index, index_adjusted):
        del lst[index_adjusted]

    def to_json(self):
        #TODO
        pass

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (self.__class__.__name__, self.name, params_str, self.deterministic)

class Sequential(Augmenter, list):
    def __init__(self, children=None, random_order=False, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        list.__init__(self, children if children is not None else [])
        #self.children = children if children is not None else []
        self.random_order = random_order

    def _transform(self, images, random_state):
        if self.random_order:
            result = images
            #for augmenter in self.children:
            for index in random_state.permute(len(self)):
                result = self[index].transform(result)
            return result
        else:
            result = images
            #for augmenter in self.children:
            for augmenter in self:
                result = augmenter.transform(result)
            return result

    def _to_deterministic(self):
        #augs = [aug.to_deterministic() for aug in self.children]
        augs = [aug.to_deterministic() for aug in self]
        seq = copy.copy(self)
        seq.children = augs
        seq.random_state = ia.new_random_state()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return []

    def add(self, augmenter):
        self.append(augmenter)

    #def append(self, augmenter):
    #    self.children.append(augmenter)
    #    return self

    #def extend(self, augmenters):
    #    self.children.extend(augmenters)
    #    return self

    def get_children_lists(self):
        return self

    def __str__(self):
        #augs_str = ", ".join([aug.__str__() for aug in self.children])
        augs_str = ", ".join([aug.__str__() for aug in self])
        return "AugmenterSequence(name=%s, augmenters=[%s], deterministic=%s)" % (self.name, augs_str, self.deterministic)

class Sometimes(Augmenter):
    def __init__(self, p, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        if isinstance(p, (float, int)) and 0 <= p <= 1:
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected float/int in range [0, 1] or StochasticParameter as p, got %s." % (type(p),))

        if then_list is None:
            self.then_list = None
        elif isinstance(then_list, Augmenter):
            self.then_list = then_list
        elif isinstance(then_list, (list, tuple)):
            self.then_list = Sequential(then_list, name="%s-then" % (self.name,), random_state=ia.new_random_state())
        else:
            raise Exception("Expected None, Augmenter or list/tuple as then_list, got %s." % (type(then_list),))

        if else_list is None:
            self.else_list = None
        elif isinstance(else_list, Augmenter):
            self.else_list = else_list
        elif isinstance(else_list, (list, tuple)):
            self.else_list = Sequential(else_list, name="%s-else" % (self.name,), random_state=ia.new_random_state())
        else:
            raise Exception("Expected None, Augmenter or list/tuple as else_list, got %s." % (type(else_list),))

    def _transform(self, images, random_state):
        result = images
        nb_images = images.shape[0]
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            subimage = images[i]
            subimage = subimage[np.newaxis, ...] # convert image to batch
            if samples[i] == 1:
                result[i] = self.if_list._transform(subimage)[0]
            else:
                result[i] = self.else_list._transform(subimage)[0]
        return result

    """
    def _to_deterministic(self, n):
        seqs = []
        then_lists = self.then_list.to_deterministic(n)
        else_lists = self.else_list.to_deterministic(n)
        for i in xrange(n):
            seqs.append(Sometimes(Deterministic(samples[i]), then_list=then_lists[i], else_list=else_lists[i], name=self.name))
        return seqs
    """
    def _to_deterministic(self):
        aug = copy.copy(self)
        aug.then_list = aug.then_list.to_deterministic()
        aug.else_list = aug.else_list.to_deterministic()
        aug.deterministic = True
        aug.random_state = ia.new_random_state()
        return aug

    def get_parameters(self):
        return [self.p]

    def get_children_lists(self):
        return [self.then_list, self.else_list]

    def __str__(self):
        return "Sometimes(p=%s, name=%s, then_list=[%s], else_list=[%s], deterministic=%s)" % (self.p, self.name, self.then_list, self.else_list, self.deterministic)

class Noop(Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

    def _transform(self, images, random_state):
        return images

    def get_parameters(self):
        return []

class Lambda(Augmenter):
    def __init__(self, func, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        self.func = func

    def _transform(self, images, random_state):
        images = func(images, random_state)
        return images

    def get_parameters(self):
        return []

def AssertLambda(func, name=None, deterministic=False, random_state=None):
    def func_assert(images, random_state):
        assert func(images)
        return images
    if name is None:
        name = "UnnamedAssertLambda"
    return Lambda(func_assert, name=name, deterministic=deterministic, random_state=random_state)

def AssertShape(shape, name=None, deterministic=False, random_state=None):
    assert len(shape) == 4, "Expected shape to have length 4, got %d with shape: %s." % (len(shape), str(shape))

    def func(images, random_state):
        assert len(images.shape) == 4, "Expected image's shape to have length 4, got %d with shape: %s." % (len(images.shape), str(images.shape))
        for i in range(4):
            expected = shape[i]
            observed = images.shape[i]
            if expected is not None:
                if isinstance(expected, int):
                    assert observed == expected, "Expected dim %d to have value %d, got %d." % (i, expected, observed)
                elif isinstance(expected, tuple):
                    assert len(expected) == 2
                    assert expected[0] <= observed < expected[1], "Expected dim %d to have value in range [%d, %d), got %d." % (i, expected[0], expected[1], observed)
                elif isinstance(expected, list):
                    assert any([observed == val for val in expected]), "Expected dim %d to have any value of %s, got %d." % (i, str(expected), observed)
                else:
                    raise Exception("Invalid datatype for shape entry %d, expected each entry to be an integer, a tuple (with two entries) or a list, got %s." % (type(expected),))
        return images

    if name is None:
        name = "UnnamedAssertShape"

    return Lambda(func, name=name, deterministic=deterministic, random_state=random_state)

class Fliplr(Augmenter):
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if isinstance(p, float):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be float or StochasticParameter, got %s." % (type(p),))

    def _transform(self, images, random_state):
        result = images
        nb_images = images.shape[0]
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            if samples[i] == 1:
                result[i] = np.fliplr(images[i])
        return result

    def get_parameters(self):
        return [self.p]

class Flipud(Augmenter):
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if isinstance(p, float):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be float or StochasticParameter, got %s." % (type(p),))

    def _transform(self, images, random_state):
        result = images
        nb_images = images.shape[0]
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            if samples[i] == 1:
                result[i] = np.flipud(images[i])
        return result

    def get_parameters(self):
        return [self.p]

class GaussianBlur(Augmenter):
    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        if isinstance(sigma, (float, int)):
            self.sigma = Deterministic(sigma)
        elif isinstance(sigma, (tuple, list)):
            assert len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(sigma)),)
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

    def _transform(self, images, random_state):
        result = images
        nb_images = images.shape[0]
        nb_channels = images.shape[3]
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            sig = samples[i]
            if sig > 0:
                for channel in range(nb_channels):
                    result[i, :, :, channel] = ndimage.gaussian_filter(result[i, :, :, channel], sig)
        return result

    def get_parameters(self):
        return [self.sigma]

class AdditiveGaussianNoise(Augmenter):
    def __init__(self, loc=0, scale=0, clip=True, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if isinstance(loc, float):
            self.loc = Deterministic(loc)
        elif isinstance(loc, (tuple, list)):
            assert len(loc) == 2, "Expected tuple/list with 2 entries for argument 'loc', got %d entries." % (str(len(scale)),)
            self.loc = Uniform(loc[0], loc[1])
        elif isinstance(loc, StochasticParameter):
            self.loc = loc
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter for argument 'loc'. Got %s." % (type(loc),))

        if isinstance(scale, float):
            self.scale = Deterministic(scale)
        elif isinstance(scale, (tuple, list)):
            assert len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (str(len(scale)),)
            self.scale = Uniform(scale[0], scale[1])
        elif isinstance(scale, StochasticParameter):
            self.scale = scale
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

        self.clip = clip

    def _transform(self, images, random_state):
        result = images
        nb_images = images.shape[0]
        nb_channels = images.shape[3]
        samples_seeds = ia.copy_random_state(random_state).randint(0, 10**6, size=(nb_images,))
        samples_loc = self.loc.draw_samples(nb_images, random_state=ia.copy_random_state(random_state))
        samples_scale = self.scale.draw_samples(nb_images, random_state=ia.copy_random_state(random_state))
        for i in xrange(nb_images):
            sample_seed = samples_seeds[i]
            sample_loc = samples_loc[i]
            sample_scale = samples_scale[i]
            assert sample_scale >= 0
            if sample_loc != 0 or sample_scale > 0:
                rs = np.random.RandomState(sample_seed)
                noise = rs.normal(sample_loc, sample_scale, size=images[i].shape)
                result[i] += (255 * noise)
        if self.clip:
            np.clip(result, 0, 255, out=result)

        return result

    def get_parameters(self):
        return [self.loc, self.scale]

class MultiplicativeGaussianNoise(Augmenter):
    # todo
    pass

class ReplacingGaussianNoise(Augmenter):
    # todo
    pass

class Dropout(Augmenter):
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if isinstance(p, float):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be float or StochasticParameter, got %s." % (type(p),))

    def _transform(self, images, random_state):
        result = images
        nb_images, height, width, nb_channels = images.shape
        samples_seeds = random_state.randint(0, 10**6, size=(nb_images,))
        for i in range(nb_images):
            seed = samples_seeds[i]
            rs_image = np.random.RandomState(seed)
            samples = self.p.draw_samples((height, width, nb_channels), random_state=rs_image)
            result[i] = result[i] * samples
        return result

    def get_parameters(self):
        return [self.p]

class Multiply(Augmenter):
    def __init__(self, mul=1.0, clip=True, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        if isinstance(mul, float):
            assert mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,)
            self.mul = Deterministic(mul)
        elif isinstance(mul, (tuple, list)):
            assert len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(mul)),)
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))
        self.clip = clip

    def _transform(self, images, random_state):
        result = images
        nb_images = result.shape[0]
        samples = self.mul.draw_samples((nb_images,), random_state=random_state)
        result = result * samples
        if self.clip:
            np.clip(result, 0, 255, out=result)
        return result

    def get_parameters(self):
        return [self.mul]

class Affine(Augmenter):
    def __init__(self, scale=1.0, translate=0, rotate=0.0, shear=0.0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        self.warp_args = warp_args if warp_args is not None else dict()

        # scale
        # float | (float, float) | [float, float] | StochasticParameter
        def scale_handle_param(param, allow_dict):
            if isinstance(param, StochasticParameter):
                return param
            elif isinstance(param, float):
                assert param > 0.0, "Expected scale to have range (0, inf), got value %.4f." % (param,)
                return Deterministic(param)
            elif isinstance(param, (tuple, list)):
                assert len(param) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                assert param[0] > 0.0 and param[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1])
                return Uniform(param[0], param[1])
            elif allow_dict and isinstance(param, dict):
                assert "x" in param or "y" in param
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else y
                y = y if y is not None else x

                return (scale_handle_param(x, Fale), scale_handle_param(y, False))
            else:
                raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))
        self.scale = scale_handle_param(scale, True)

        # translate
        # float | int | (float, float) | (int, int) | [float, float] | [int, int] | StochasticParameter
        def translate_handle_param(param, allow_dict):
            if isinstance(param, float):
                assert param > 0.0, "Expected translate to have range (0, inf), got value %.4f." % (param,)
                self.param = Deterministic(param)
            elif isinstance(param, int):
                self.param = Deterministic(param)
            elif isinstance(param, (tuple, list)):
                assert len(param) == 2, "Expected translate tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                types_unique = set([type(val) for val in param])
                assert len(types_unique) == 1, "Expected translate tuple/list to have either int or float datatype, got %s." % (str(types_unique),)
                assert types_unique in ["int", "float"], "Expected translate tuple/list to have either int or float datatype, got %s." % (str(types_unique),)

                if types_unique[0] == "int":
                    self.translate = DiscreteUniform(param[0], param[1])
                else: # float
                    assert param[0] > 0.0 and param[1] > 0.0, "Expected translate tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1])
                    self.translate = Uniform(param[0], param[1])
            elif allow_dict and isinstance(parm, dict):
                assert "x" in param or "y" in param
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else y
                y = y if y is not None else x

                return (translate_handle_param(x, Fale), translate_handle_param(y, False))
            elif isinstance(param, StochasticParameter):
                self.translate = param
            else:
                raise Exception("Expected float, or int or tuple/list with 2 entries of both floats or ints or StochasticParameter. Got %s." % (type(param),))
        self.translate = translate_handle_param(translate, True)

        # rotate
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(rotate, StochasticParameter):
            self.rotate = rotate
        elif isinstance(rotate, (float, int)):
            self.rotate = rotate
        elif isinstance(rotate, (tuple, list)):
            assert len(rotate) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (str(len(rotate)),)
            types = [type(r) for r in rotate]
            assert all([val in ["float", "int"] for val in types]), "Expected floats/ints in rotate tuple/list, got %s." % (str(types),)
            self.rotate = Uniform(rotate[0], rotate[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))

        # shear
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif isinstance(shear, (float, int)):
            self.shear = shear
        elif isinstance(shear, (tuple, list)):
            assert len(shear) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (str(len(shear)),)
            types = [type(r) for r in rotate]
            assert all([val in ["float", "int"] for val in types]), "Expected floats/ints in shear tuple/list, got %s." % (str(types),)
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))

    def _transform(self, images, random_state):
        # skimage's warp() converts to 0-1 range, so we use float here and then convert
        # at the end
        result = images.astype(np.float32, copy=False)

        nb_images, height, width = images.shape[0], images.shape[1], images.shape[2]

        scale_samples, translate_samples_px, rotate_samples, shear_samples = self._draw_samples(nb_images, random_state)

        shift_x = int(width / 2.0)
        shift_y = int(height / 2.0)

        for i in xrange(nb_images):
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x_px, translate_y_px = translate_samples_px[0][i], translate_samples_px[1][i]
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x, translate_y),
                    rotation=rotate,
                    shear=shear
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center).inverse
                result[i, ...] = tf.warp(result[i, ...], matrix, **self.warp_args)

        random_state.randint(0, 10**6) # move random state forward

        return (result * 255.0).astype(np.uint8, copy=False)

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear]

    def _draw_samples(self, nb_samples, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 10)),
                self.scale[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 20)),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 30))
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 40)),
                self.translate[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 50)),
            )
        else:
            translate_samples = self.translate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 60))
            translate_samples = (translate_samples, translate_samples)

        assert translate_samples[0].dtype in [np.int32, np.int64, np.float32, np.float64]
        assert translate_samples[1].dtype in [np.int32, np.int64, np.float32, np.float64]
        translate_samples_px = [None, None]
        if translate_samples[0].dtype in [np.float32, np.float64]:
            translate_samples_px[0] = translate_samples[0] * width
        else:
            translate_samples_px[0] = translate_samples[0]
        if translate_samples[1].dtype in [np.float32, np.float64]:
            translate_samples_px[1] = translate_samples[1] * height
        else:
            translate_samples_px[1] = translate_samples[1]

        rotate_samples = self.rotate.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 70))
        shear_samples = self.shear.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 80))

        return scale_samples, translate_samples_px, rotate_samples, shear_samples

class ElasticTransformation(Augmenter):
    # todo
    pass
