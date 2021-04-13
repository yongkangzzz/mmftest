from shap.plots._text import *

def text(shap_values, num_starting_labels=0, group_threshold=1, separator='', xmin=None, xmax=None, cmax=None):
    """ Plots an explanation of a string of text using coloring and interactive labels.
    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.
    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap values for a string(# input_tokens x output_tokens).
    num_starting_labels : int
        Number of tokens (sorted in decending order by corresponding SHAP values) that are uncovered in the initial view. When set to 0 all tokens
        covered. 
    group_threshold : float
        The threshold used to group tokens based on interaction affects of SHAP values.
    separator : string
        The string seperator that joins tokens grouped by interation effects and unbroken string spans.
    xmin : float
        Minimum shap value bound. 
    xmax : float
        Maximum shap value bound.
    cmax : float
        Maximum absolute shap value for sample. Used for scaling colors for input tokens. 
    """
    from IPython.core.display import display, HTML

    def values_min_max(values, base_values):
        """ Used to pick our axis limits.
        """
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2 and (shap_values.output_names is None or isinstance(shap_values.output_names, str)):
        # NOTE output for recursion
        final_out = ""

        xmin = 0
        xmax = 0
        cmax = 0

        for i in range(0, len(shap_values)):

            values, clustering = unpack_shap_explanation_contents(shap_values[i])
            tokens, values, group_sizes = process_shap_values(shap_values[i].data, values, group_threshold, separator, clustering)

            if i == 0:
                xmin, xmax, cmax = values_min_max(values, shap_values[i].base_values)
                continue

            xmin_i,xmax_i,cmax_i = values_min_max(values, shap_values[i].base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        for i in range(len(shap_values)):
            # display(HTML("<br/><b>"+ordinal_str(i)+" instance:</b><br/>"))
            # text(shap_values[i], num_starting_labels=num_starting_labels, group_threshold=group_threshold, separator=separator, xmin=xmin, xmax=xmax, cmax=cmax)
            final_out += ("<br/><b>"+ordinal_str(i)+" instance:</b><br/>")
            final_out += text(shap_values[i], num_starting_labels=num_starting_labels, group_threshold=group_threshold, separator=separator, xmin=xmin, xmax=xmax, cmax=cmax)
        return final_out

    elif len(shap_values.shape) == 2 and shap_values.output_names is not None:
        text_to_text(shap_values)
        return
    elif len(shap_values.shape) == 3:
        for i in range(len(shap_values)):
            display(HTML("<br/><b>"+ordinal_str(i)+" instance:</b><br/>"))
            text(shap_values[i])
        return


    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(shap_values.values, shap_values.base_values)
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new


    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(shap_values.data, values, group_threshold, separator, clustering)
    
    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    maxv = values.max()
    minv = values.min()
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())
    
    uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [t.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') for t in tokens]
    out += svg_force_plot(values, shap_values.base_values, shap_values.base_values + values.sum(), encoded_tokens, uuid, xmin, xmax)
    
    for i in range(len(tokens)):
        scaled_value = 0.5 + 0.5 * values[i] / cmax
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0]*255, color[1]*255, color[2]*255, color[3])
        
        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"
        
        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])
        
        # the HTML for this token
        out += "<div style='display: " + wrapper_display + "; text-align: center;'>" \
             + "<div style='display: " + label_display + "; color: #999; padding-top: 0px; font-size: 12px;'>" \
             + value_label \
             + "</div>" \
             + f"<div id='_tp_{uuid}_ind_{i}'" \
             +   "style='display: inline; background: rgba" + str(color) + "; border-radius: 3px; padding: 0px'" \
             +   "onclick=\"if (this.previousSibling.style.display == 'none') {" \
             +       "this.previousSibling.style.display = 'block';" \
             +       "this.parentNode.style.display = 'inline-block';" \
             +     "} else {" \
             +       "this.previousSibling.style.display = 'none';" \
             +       "this.parentNode.style.display = 'inline';" \
             +     "}" \
             +   "\"" \
             +   f"onmouseover=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;" \
             +   "\"" \
             +   f"onmouseout=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;" \
             +   "\"" \
             + ">" \
             + tokens[i].replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') \
             + "</div>" \
             + "</div>"

    # display(HTML(out))
    return out