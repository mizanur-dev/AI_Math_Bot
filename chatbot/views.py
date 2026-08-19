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


def sanitize_latex_to_plain(text):
    """Convert LaTeX math to plain readable text so frontends without
    a LaTeX renderer (KaTeX/MathJax) can display the response cleanly."""

    # Remove display math delimiters $$...$$
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
    # Remove inline math delimiters $...$
    text = re.sub(r'\$(.+?)\$', r'\1', text)

    # Common LaTeX commands → plain text / Unicode
    text = text.replace('\\times', '×')
    text = text.replace('\\div', '÷')
    text = text.replace('\\pm', '±')
    text = text.replace('\\mp', '∓')
    text = text.replace('\\leq', '≤')
    text = text.replace('\\geq', '≥')
    text = text.replace('\\neq', '≠')
    text = text.replace('\\approx', '≈')
    text = text.replace('\\infty', '∞')
    text = text.replace('\\pi', 'π')
    text = text.replace('\\alpha', 'α')
    text = text.replace('\\beta', 'β')
    text = text.replace('\\theta', 'θ')
    text = text.replace('\\sum', '∑')
    text = text.replace('\\int', '∫')
    text = text.replace('\\rightarrow', '→')
    text = text.replace('\\leftarrow', '←')
    text = text.replace('\\Rightarrow', '⇒')
    text = text.replace('\\cdot', '·')
    text = text.replace('\\ldots', '...')
    text = text.replace('\\%', '%')
    text = text.replace('\\$', '$')
    text = text.replace('\\\\', '\\')

    # \frac{a}{b} → (a)/(b)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)
    # \sqrt{x} → √(x)
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', text)
    # \text{...} → just the content
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    # \left and \right delimiters → just the bracket
    text = re.sub(r'\\left([(\[{|])', r'\1', text)
    text = re.sub(r'\\right([)\]}|])', r'\1', text)
    # Remove remaining \commandname (single backslash commands without braces)
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    # Clean up extra braces used for LaTeX grouping
    text = text.replace('{', '').replace('}', '')

    return text.strip()


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
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured. Please set it in the .env file.")
            cls._llm_instance = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.5,
                max_output_tokens=1024,
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
        
        # Load last 6 messages (3 exchanges) for context while saving tokens
        history_data = session_data.get(chat_history_key, [])
        recent_history = history_data[-6:] if len(history_data) > 6 else history_data
        
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
            return Response(
                {"detail": f"AI service unavailable: {str(e)}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Concise system prompt — formatting is handled by post-processing
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a math assistant. Solve math problems with clear, concise step-by-step solutions. Respond in the user's language. Keep answers brief: show essential steps and the final answer only. For non-math queries, politely say you only help with math. Do not use markdown bold/italic formatting."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        chain = prompt | llm
        
        # Invoke chain directly with history messages (no deprecated RunnableWithMessageHistory)
        try:
            response = chain.invoke({"input": user_message, "history": history_messages})
            ai_response = response.content

            # Post-process: strip markdown bold/italic that the LLM may still produce
            ai_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', ai_response)  # Remove **bold**
            ai_response = re.sub(r'(?<!\\)\*([^*]+)\*', r'\1', ai_response)  # Remove *italic*

            # Convert LaTeX math to plain readable text
            ai_response = sanitize_latex_to_plain(ai_response)
        except Exception as e:
            return Response({"detail": f"Chat service error: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)
        
        # Save updated history - keep reasonable amount for production level use
        full_history = session_data.get(chat_history_key, [])
        full_history.append({'type': 'human', 'content': user_message})
        full_history.append({'type': 'ai', 'content': ai_response})
        
        # Limit history to last 6 messages for optimal performance and token savings
        if len(full_history) > 6:
            full_history = full_history[-6:]
        
        # Update session
        session_data[chat_history_key] = full_history
        session_obj.session_data = Session.objects.encode(session_data)
        session_obj.save()

        response_serializer = ChatResponseSerializer({'response': ai_response})
        return Response(response_serializer.data, status=status.HTTP_200_OK)