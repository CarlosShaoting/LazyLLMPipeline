from ...base_data import DataOperatorRegistry


@DataOperatorRegistry.register(one_item=False, tag='all')
def build_pre_suffix(data, input_key='content', prefix='', suffix=''):
    assert isinstance(data, list)
    for item in data:
        item[input_key] = f'{prefix}{item.get(input_key, "")}{suffix}'
    return data



