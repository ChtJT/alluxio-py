import lance

ds = lance.dataset("/private/var/folders/xc/3rwstl_x16qg0h9svt0s2w_00000gn/T/pytest-of-cjanus/pytest-30/test_text_converter_on_repo_fi1/converted/alpaca_farm_human_crossannotations.lance")

print(ds.scanner().to_table())
