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
            return Response(
                {"detail": f"AI service unavailable: {str(e)}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Professional prompt for comprehensive math assistance
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional AI Mathematics Assistant designed to provide comprehensive, accurate, and well-explained solutions to mathematical problems.

Your responsibilities:
- Provide detailed step-by-step explanations for mathematical problems
- Show all working steps clearly and logically
- Explain concepts when necessary to aid understanding
- Always respond in the same language as the user's query
- Cover topics including: Algebra, Calculus, Geometry, Statistics, Trigonometry, Linear Algebra, Discrete Mathematics, and more

MATH FORMATTING RULES (CRITICAL - you MUST follow these exactly):
- For ALL mathematical expressions, variables, numbers in equations, and symbols, you MUST use LaTeX wrapped in dollar sign delimiters.
- Use single dollar signs $...$ for inline math expressions within a sentence. Example: The result is $x = 5$.
- Use double dollar signs $$...$$ for display math (standalone equations on their own line). Example:
$$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$$
- Use LaTeX commands for all math operations: \\frac{{}}{{}}, \\times, \\div, \\sqrt{{}}, \\pm, \\leq, \\geq, \\neq, \\sum, \\int, \\infty, etc.
- For percentages, write $17\\%$ not 17%.
- For currency values mentioned in math context, write $\\$10.00$ for inline.
- NEVER write raw LaTeX commands without dollar sign delimiters.
- NEVER use \\[ ... \\] or \\( ... \\) delimiters. ONLY use $...$ and $$...$$.

TEXT FORMATTING RULES:
- Use plain text for step labels: Step 1: , Step 2: , etc. Do NOT use asterisks or any bold/italic markdown syntax (no ** or * around text).
- Use blank lines between steps for readability.
- Use numbered lists (1. 2. 3.) or dashes (- ) when listing items.
- Separate explanation text from math equations clearly.
- Keep plain text (non-math) outside of dollar sign delimiters.
- NEVER use markdown bold (**text**) or italic (*text*) formatting. All non-math text must be plain text only.

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

        response_serializer = ChatResponseSerializer({'response': ai_response})
        return Response(response_serializer.data, status=status.HTTP_200_OK)