# Relax test condition for unlucky seed choice 

After trying back and forth I came to the conclusion that the 
`test/augmenters/test_geometric.py:_test_image_cbaoi_alignment`fails
sindce target value is highly dependend on the seed value. With `seed=1` for `iaa.ElasticTransformation` `count_bad` goes to 0.

This provides the conclusion even with a changed seed it cannot be granted that the test will pass for future random generators. Thus
a relaxed target value might be the better choice.