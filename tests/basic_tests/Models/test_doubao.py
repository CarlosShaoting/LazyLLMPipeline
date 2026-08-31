from lazyllm.module.llms.onlinemodule.supplier import doubao as doubao_supplier
from lazyllm.module.llms.onlinemodule.supplier.doubao import DoubaoText2Image, DoubaoText2Video


class _Response:
    def __init__(self, data=None, content=b''):
        self._data = data
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_doubao_image_uses_ark_rest_api(monkeypatch):
    captured = []

    def fake_request(method, url, **kwargs):
        captured.append((method, url, kwargs))
        return _Response({'data': [
            {'url': 'https://example.com/image-1.png'},
            {'url': 'https://example.com/image-2.png'},
        ]})

    def fake_get(url, **kwargs):
        return _Response(content=f'image:{url}'.encode())

    monkeypatch.setattr(doubao_supplier.requests, 'request', fake_request)
    monkeypatch.setattr(doubao_supplier.requests, 'get', fake_get)
    monkeypatch.setattr(doubao_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(doubao_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    image = DoubaoText2Image(api_key='test-key')
    result = image._forward(input='a red panda', n=2)

    method, url, request = captured[0]
    assert method == 'POST'
    assert url == 'https://ark.cn-beijing.volces.com/api/v3/images/generations'
    assert request['headers']['Authorization'] == 'Bearer test-key'
    assert request['json']['model'] == DoubaoText2Image.MODEL_NAME
    assert request['json']['prompt'] == 'a red panda'
    assert request['json']['sequential_image_generation'] == 'auto'
    assert request['json']['sequential_image_generation_options'] == {'max_images': 2}
    assert result == [
        b'image:https://example.com/image-1.png',
        b'image:https://example.com/image-2.png',
    ]


def test_doubao_video_uses_ark_rest_api(monkeypatch):
    captured = []
    responses = iter([
        _Response({'id': 'task-123'}),
        _Response({'id': 'task-123', 'status': 'running'}),
        _Response({
            'id': 'task-123',
            'status': 'succeeded',
            'content': {'video_url': 'https://example.com/video.mp4'},
        }),
    ])

    def fake_request(method, url, **kwargs):
        captured.append((method, url, kwargs))
        return next(responses)

    monkeypatch.setattr(doubao_supplier.requests, 'request', fake_request)
    monkeypatch.setattr(doubao_supplier.requests, 'get', lambda *args, **kwargs: _Response(content=b'video-bytes'))
    monkeypatch.setattr(doubao_supplier.time, 'sleep', lambda _: None)
    monkeypatch.setattr(doubao_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(doubao_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = DoubaoText2Video(api_key='test-key')
    result = video._forward(input='a bird flying', poll_interval=0)

    assert [item[:2] for item in captured] == [
        ('POST', 'https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks'),
        ('GET', 'https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/task-123'),
        ('GET', 'https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/task-123'),
    ]
    assert captured[0][2]['json'] == {
        'model': DoubaoText2Video.MODEL_NAME,
        'content': [{
            'type': 'text',
            'text': ('a bird flying --resolution 480p --duration 2 --ratio 16:9 '
                     '--camerafixed false --watermark true'),
        }],
    }
    assert result == [b'video-bytes']
