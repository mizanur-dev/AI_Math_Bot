from rest_framework import status
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError, NotFound
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from .serializers import ChatRequestSerializer, ChatResponseSerializer, EmailSerializer
from django.conf import settings
from django.contrib.sessions.models import Session
import uuid
import re



class EmailView(CreateAPIView):
    serializer_class = EmailSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        email = serializer.validated_data['email']
        
        # Generate unique session ID based on email
        session_id = f"{email}_{uuid.uuid4().hex[:8]}"
        
        # Create a new session
        request.session.create()
        
        # Store email and our custom session ID
        request.session['user_email'] = email
        request.session['custom_session_id'] = session_id
        
        return Response({
            "message": f"Email set successfully: {email}. You can now use the chatbot.",
            "session_id": session_id
        }, status=status.HTTP_200_OK)

class ChatView(CreateAPIView):
    serializer_class = ChatRequestSerializer
    
    # Initialize LLM once at class level for better performance
    _llm_instance = None
    
    @classmethod
    def get_llm(cls):
        if cls._llm_instance is None:
            cls._llm_instance = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.5,
            )
        return cls._llm_instance

    def get_session_by_custom_id(self, session_id):
        """Find session by our custom session ID - optimized"""
        # Cache session lookup to avoid repeated database queries
        sessions = Session.objects.all()
        for session in sessions:
            session_data = session.get_decoded()
            if session_data.get('custom_session_id') == session_id:
                return session, session_data
        return None, None

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user_message = serializer.validated_data['message']
        session_id = serializer.validated_data['session_id']
        
        if not session_id:
            raise ValidationError("Session ID is required")
        
        # Find session by custom session ID
        session_obj, session_data = self.get_session_by_custom_id(session_id)
        
        if not session_obj or not session_data:
            raise NotFound("Invalid session ID. Please set your email first via /api/set_email/")
        
        
        if 'user_email' not in session_data:
            raise NotFound("Invalid session. Please set your email first via /api/set_email/")
        
        # Chat processing logic
        chat_history_key = f'chat_history_{session_id}'
        
        # Load last 10 messages (5 exchanges) for better context
        history_data = session_data.get(chat_history_key, [])
        recent_history = history_data[-10:] if len(history_data) > 10 else history_data
        
        # Build history as LangChain message objects
        history_messages = []
        for item in recent_history:
            if item['type'] == 'human':
                history_messages.append(HumanMessage(content=item['content']))
            elif item['type'] == 'ai':
                history_messages.append(AIMessage(content=item['content']))

        # Use cached LLM instance
        try:
            llm = self.get_llm()
        except Exception as e:
            raise ValidationError(f"LLM initialization failed: {str(e)}")
        
        # Professional prompt for comprehensive math assistance
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional AI Mathematics Assistant designed to provide comprehensive, accurate, and well-explained solutions to mathematical problems.

Your responsibilities:
- Provide detailed step-by-step explanations for mathematical problems
- Show all working steps clearly and logically
- Use proper mathematical notation and terminology
- Explain concepts when necessary to aid understanding
- Always respond in the same language as the user's query
- Cover topics including: Algebra, Calculus, Geometry, Statistics, Trigonometry, Linear Algebra, Discrete Mathematics, and more

For non-mathematical queries:
- Politely inform the user that you specialize in mathematics only
- Suggest they ask a math-related question instead

Maintain a professional, helpful, and educational tone in all responses."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        chain = prompt | llm
        
        # Invoke chain directly with history messages (no deprecated RunnableWithMessageHistory)
        try:
            response = chain.invoke({"input": user_message, "history": history_messages})
            ai_response = response.content
        except Exception as e:
            return Response({"detail": f"Chat service error: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)
        
        # Save updated history - keep reasonable amount for production level use
        full_history = session_data.get(chat_history_key, [])
        full_history.append({'type': 'human', 'content': user_message})
        full_history.append({'type': 'ai', 'content': ai_response})
        
        # Limit history to last 10 messages for optimal performance and context
        if len(full_history) > 10:
            full_history = full_history[-10:]
        
        # Update session
        session_data[chat_history_key] = full_history
        session_obj.session_data = Session.objects.encode(session_data)
        session_obj.save()
        # Clean up Markdown-style emphasis (asterisks/underscores) so responses read naturally
        def _strip_markdown_emphasis(text: str) -> str:
            if not text:
                return text
            # Remove bold/italic markers like **bold**, __bold__, *italic*, _italic_
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.S)
            text = re.sub(r"__(.*?)__", r"\1", text, flags=re.S)
            text = re.sub(r"\*(\w.*?)\*", r"\1", text, flags=re.S)
            text = re.sub(r"_(\w.*?)_", r"\1", text, flags=re.S)
            # Remove leading list bullets (lines starting with '* ' or '- ')
            text = re.sub(r"(?m)^\s*[\*\-]\s+", "", text)
            return text

        cleaned_response = _strip_markdown_emphasis(ai_response)
        response_serializer = ChatResponseSerializer({'response': cleaned_response})
        return Response(response_serializer.data, status=status.HTTP_200_OK)