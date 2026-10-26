"""Conservative boundary for named choices; not a general language parser.

Only a small grammar of unconditional named choices may bypass policy checks.
Any additional wording alongside a known model name requires clarification.
This guard is independent of the LLM's chosen route and choice_quote.
"""
import re

ALIAS = {
    'acoustic_lr': ('acoustic_lr', 'logistic regression', 'lr'),
    'acoustic_rf': ('acoustic_rf', 'random forest', 'rf'),
    'spectrogram_resnet18_linear': ('spectrogram_resnet18_linear', 'resnet18', 'resnet'),
}
MODEL = '|'.join(re.escape(a) for values in ALIAS.values() for a in values)
CONTEXT = r'(?:telephone(?: vowel)?|italian(?: vowel)?)(?: data| dataset| set)?'
DIRECT = re.compile(
    r'(?:please\s+)?(?:use|select|choose|run|execute|i choose|i select|i want to (?:use|run))\s+'
    r'(?:the\s+)?(?P<model>'+MODEL+r')(?:\s+(?:model|classifier|pipeline))?'
    r'(?:\s+for\s+(?:the\s+)?'+CONTEXT+r')?[.!]?', re.I)
EVALUATE = re.compile(r'please evaluate the '+CONTEXT+r' using (?P<model>'+MODEL+r')[.!]?',re.I)
BARE = re.compile(r'(?P<model>'+MODEL+r')[.!]?',re.I)
MENTION = re.compile(r'(?<!\w)(?:'+MODEL+r')(?!\w)',re.I)
QUESTION = ('A named model with additional conditions or unsupported wording cannot '
            'be treated as an unconditional choice. Choose an unconditional model, '
            'or state a performance policy without naming a model. No model was executed.')

def unconditional_choice(request):
    text=' '.join(request.strip().split())
    match=next((m for pattern in (DIRECT,EVALUATE,BARE) if (m:=pattern.fullmatch(text))),None)
    if match is None:return None
    word=match['model'].lower()
    return next(k for k,v in ALIAS.items() if word in v)

def check_chat_boundary(request):
    chosen=unconditional_choice(request)
    if MENTION.search(request) and chosen is None:
        return {'status':'needs_clarification','selected_model':None,'question':QUESTION,
                'boundary_version':'1.0','executed':0}
    return {'status':'allowed','unconditional_model':chosen,'boundary_version':'1.0'}

def validate_choice_intent(request,model):
    if unconditional_choice(request)!=model:
        raise ValueError('Explicit choice is not an accepted unconditional request; '+QUESTION)

def require_requirements_only(request):
    if MENTION.search(request):
        raise ValueError('Named-model requests are unsupported in the requirements route; '+QUESTION)
