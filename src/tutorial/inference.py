import math
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from torch.nn import functional as F
import torch

def show_predictions(model, tokenizer, device, text):
    for i, line in enumerate(text):
        print(f"line: {line}")
        model = model.to(device)
        logits, label_ids = predict_text(model, tokenizer, line, device=device)
            
        show_colorized_tokens(
            tokenizer=tokenizer,
            logits=logits,
            label_ids=label_ids,
            
            # the metric function to use
            metric_fn=causal_loss_metric,
        
            # how to translate metric to colors
            color_encoder=ColorEncoder(
                is_relative=True,
                #lower_bound=0,
                #upper_bound=10,
                cmap='plasma',
            ),
            top_k = 10,
            
            # Filter threshold on metric
            #threshold=0,
            pad_lines=15
        )
        print("\n")

def predict_text(model, tokenizer, input_text, device):
    batch_encoding = tokenizer(
        input_text,
        truncation=True,
        return_tensors='pt',
        verbose=True,
    )
    input_ids = batch_encoding['input_ids'].to(device)
    model.eval()
    model.to(device)
    logits = model(input_ids=input_ids)
    return logits.cpu().detach().float(), input_ids.cpu().detach()

def show_colorized_tokens(tokenizer, logits, label_ids, metric_fn, color_encoder, top_k, threshold=None, pad_lines=20):
    metrics, metric_min, metric_max, metric_label = metric_fn(logits=logits, label_ids=label_ids)
    value_min, value_max = metrics.aminmax()
    
    print(f"Metric '{metric_label}': n={metrics.numel()}, min={value_min}, max={value_max}, mean={metrics.mean()}, range=({metric_min}, {metric_max})")

    # if causal only; we need to pad the left with something to alighn the predictions.
    metrics = torch.cat((torch.zeros(metrics.size(0), 1, device=metrics.device, dtype=metrics.dtype), metrics), dim=-1)
    if metrics.size(-1) > label_ids.size(-1):
        metrics = metrics.narrow(-1, 0, label_ids.size(-1))
    colors = color_encoder(metrics, value_min, value_max, metric_min=metric_min, metric_max=metric_max)
    html_text = tooltip_style

    label_ids = restore_pad_ids(tokenizer, label_ids)
    for i in range(label_ids.size(0)):
        metric_mean = metrics[i].mean()
        info = f"Metric[{i}] '{metric_label}': n={metrics[i].numel()}, min={metrics[i].min()}, max={metrics[i].max()}, mean={metric_mean}<br>"
        html_text += info
        if threshold is not None and metric_mean < threshold:
            continue
        token_seq = tokenizer.batch_decode(label_ids[i], skip_special_tokens=True)
        color_seq = colors[i]
        token_info_list = generate_token_info_list(tokenizer, token_seq, metric_label, metrics[i], logits[i], top_k)
        text = color_encode_html_tokens(token_seq, color_seq, token_info_list)
        html_text += text + "<br>"

    for _ in range(pad_lines):
        html_text += "<br>"
    
    display(HTML(html_text))
    
# Replace '-100' tokens with the tokenizer's 'pad' token.
def restore_pad_ids(tokenizer, label_ids):
    return torch.where(label_ids == -100, tokenizer.eos_token_id, label_ids) 

tooltip_style = """
<style>
/* Tooltip container class */
.token {
  position: relative;
  display: inline-block;
}

/* Tooltip text */
.token .tooltip {
  visibility: hidden;
  width: 300px;
  background-color: black;
  color: #fff;
  text-align: left;
  padding: 5px 0;
  border-radius: 6px;
 
  /* Position the tooltip text - see examples below! */
  position: absolute;
  z-index: 1;
}

/* Show the tooltip text when you mouse over the tooltip container */
.token:hover .tooltip {
  visibility: visible;
}
</style>
"""

def escape_html_token(token):
    match token:
        case '\n':
            return '<br>'
        case '<':
            return '&lt;'
        case '>':
            return '&gt;'
        case '"':
            return '&quot;'
        case "'":
            return '&#39;'
        case '&':
            return '&amp;'
        case _:
            return token

def html_color(color):
    return "#{:02x}{:02x}{:02x}".format(int(255*color[0]), int(255*color[1]), int(255*color[2]))

def color_encode_html_tokens(token_seq, color_seq, info_seq):
    text = ""
    for token, color, info in zip(token_seq, color_seq, info_seq):
        if token == '\n':
            text += "<br>"
        else:
            # HTML will eat your space tokens if you don't do this!
            if len(token) > 0 and token[0] == ' ':
                token = "&nbsp;" + token[1:]
            text += f"<span class='token' style='color: {html_color(color)}'>{escape_html_token(token)}<span class='tooltip'>{info}</span></span>"
    return text

class ColorEncoder:
    def __init__(self, is_relative=True, upper_bound=math.inf, lower_bound=-math.inf, cmap='viridis'):
        self.is_relative = is_relative
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.colormap = plt.get_cmap(cmap)

    def __call__(self, metrics, value_min, value_max, metric_min=None, metric_max=None):
        if self.is_relative or metric_min is None and metric_max is None:
            minimum, maximum = value_min, value_max
        elif metric_min is None:
            minimum, maximum = value_min, metric_max
        elif metric_max is None:
            minimum, maximum = metric_min, value_max
        else:
            minimum, maximum = metric_min, metric_max
            
        minimum = max(minimum, self.lower_bound)
        maximum = min(maximum, self.upper_bound)
        return self.colormap(self.normalize_metric(metrics, minimum, maximum))
        
    def normalize_metric(self, metric, minimum, maximum):
        return torch.clamp(input=(metric - minimum) / (maximum - minimum), min=0.0, max=1.0)

def topk_predicted_tokens(tokenizer, logits, top_k=5):
    top_prob, top_indices = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    top_tokens = tokenizer.batch_decode(top_indices.flatten(), skip_special_tokens=True)
        
    return top_prob.flatten(), top_tokens
        
def generate_token_info_list(tokenizer, token_seq, metric_label, metrics, logits, top_k=5):
    top_prob, top_tokens = topk_predicted_tokens(tokenizer, logits, top_k)
    info_list = []
    for j in range(len(metrics)):
        text = f"Token: '{token_seq[j]}'<br>{metric_label}: {'%.5f' % metrics[j]}<br>---------------<br>"
        if j != 0: # And is causal!
            start = (j - 1) * top_k
            for k in range(start, start + top_k):
                top_token = top_tokens[k]
                if top_token == '\n':
                    top_token = '\\n'
                text += f"{'%.2f' % top_prob[k]} : '{escape_html_token(top_token)}'<br>"
        info_list.append(text)
    return info_list

#### Logits and Lables Metric Functions
def causal_loss_metric(logits, label_ids, reduction='none'):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction=reduction)\
        .view(label_ids.size(0), label_ids.size(1) - 1)
    
    return loss, 0, None, "Causal Loss"

