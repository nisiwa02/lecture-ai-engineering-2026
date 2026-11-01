import weakref

from app import ask as ask_module


def test_chat_sends_messages_to_same_conversation(monkeypatch):
    class Response:
        text = "answer"
        usage_metadata = None

    class Conversation:
        def __init__(self):
            self.messages = []
            self.client_ref = None

        def send_message(self, message):
            assert self.client_ref() is not None
            self.messages.append(message)
            return Response()

    class Chats:
        def __init__(self):
            self.conversation = Conversation()
            self.create_calls = []

        def create(self, *, model):
            self.create_calls.append(model)
            return self.conversation

    class Client:
        def __init__(self, *, api_key):
            self.chats = Chats()

    clients = []
    chats = []

    def create_client(*, api_key):
        client = Client(api_key=api_key)
        client.chats.conversation.client_ref = weakref.ref(client)
        clients.append(weakref.ref(client))
        chats.append(client.chats)
        return client

    prompts = iter(["最初の質問", "続きの質問", ""])
    monkeypatch.setattr(ask_module.genai, "Client", create_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask_module.tools,
        "retrieve_glossary",
        lambda question: f"retrieved:{question}",
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))

    ask_module.chat()

    assert len(clients) == 1
    assert chats[0].create_calls == [ask_module.MODEL]
    assert "retrieved:最初の質問" in chats[0].conversation.messages[0]
    assert "retrieved:続きの質問" in chats[0].conversation.messages[1]