import asyncio
from chatbot import ask_assistant
from starlette.requests import Request as StarletteRequest

async def fake_request(data):
    scope = {
        'type': 'http',
        'method': 'POST',
        'path': '/ask',
        'headers': [],
        'client': ('127.0.0.1', 12345),
        'server': ('127.0.0.1', 8000),
        'scheme': 'http',
    }
    req = StarletteRequest(scope, receive=None)

    async def _json():
        return data

    req.json = _json
    return await ask_assistant(req)

if __name__ == '__main__':
    query = 'کیا حج کے احرام میں مخصوص لباس پہننا ضروری ہے؟'
    result = asyncio.run(fake_request({'query': query}))
    print('Query:', query)
    print('Result:')
    print(result)
