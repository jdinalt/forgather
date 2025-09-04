" Vim syntax file for Forgather configuration files
" Language: Forgather Configuration (YAML + Jinja2 + Custom Extensions)
" Maintainer: Claude Code
" Latest Revision: 2025-01-16

if exists("b:current_syntax")
  finish
endif

" Include base YAML syntax first
runtime! syntax/yaml.vim
unlet! b:current_syntax

" Forgather-specific syntax groups
syn region forgatherLineComment start=/^\s*##/ end=/$/ contains=@Spell
syn region forgatherInlineComment start=/\s\+##/ end=/$/ contains=@Spell

" Standard Jinja2 syntax - use keepend and contained properly
" Jinja2 comments - these don't interfere with YAML structure
syn region forgatherJinjaComment start=/{#/ end=/#}/ contains=@Spell keepend

" Jinja2 print statements {{ }} - use keepend to not interfere with YAML
syn region forgatherJinjaPrint start=/{{/ end=/}}/ keepend contains=forgatherJinjaPrintDelim,forgatherJinjaVar,forgatherJinjaString,forgatherJinjaFilter,forgatherJinjaFunction,forgatherJinjaBrackets
syn match forgatherJinjaPrintDelim /{{/ contained nextgroup=forgatherJinjaPrintContent
syn match forgatherJinjaPrintDelim /}}/ contained
syn region forgatherJinjaPrintContent start=/\S/ end=/\ze}}/ contained contains=forgatherJinjaVar,forgatherJinjaString,forgatherJinjaFilter,forgatherJinjaFunction,forgatherJinjaBrackets

" Jinja2 statement blocks {% %} - use keepend to not interfere with YAML
syn region forgatherJinjaBlock start=/{%/ end=/%}/ keepend contains=forgatherJinjaBlockDelim,forgatherJinjaBlockContent
syn match forgatherJinjaBlockDelim /{%/ contained nextgroup=forgatherJinjaBlockContent
syn match forgatherJinjaBlockDelim /%}/ contained
syn region forgatherJinjaBlockContent start=/\S/ end=/\ze%}/ contained contains=forgatherJinjaKeyword,forgatherJinjaVar,forgatherJinjaString,forgatherJinjaFunction,forgatherJinjaBrackets

" Jinja2 line statements
syn match forgatherLineStmt /^\s*--\s/ nextgroup=forgatherJinjaStmt
syn match forgatherLineStmtTrimLeft /^\s*<<\s/ nextgroup=forgatherJinjaStmt  
syn match forgatherLineStmtTrimRight /^\s*>>\s/ nextgroup=forgatherJinjaStmt
syn match forgatherPrintStmt /^\s*==\s/ nextgroup=forgatherJinjaExpr
syn match forgatherPrintStmtTrim /^\s*=>\s/ nextgroup=forgatherJinjaExpr

" Jinja2 content after line statements
syn region forgatherJinjaStmt start=/\S/ end=/$/ contained contains=forgatherJinjaKeyword,forgatherJinjaVar,forgatherJinjaString,forgatherJinjaFunction,forgatherJinjaBrackets
syn region forgatherJinjaExpr start=/\S/ end=/$/ contained contains=forgatherJinjaVar,forgatherJinjaString,forgatherJinjaFilter,forgatherJinjaFunction,forgatherJinjaBrackets

" Jinja2 keywords
syn keyword forgatherJinjaKeyword contained if endif else elif for endfor block endblock extends include set macro endmacro with endwith
syn keyword forgatherJinjaKeyword contained super loop caller varargs kwargs in

" Jinja2 variables and expressions
syn match forgatherJinjaVar contained /\<\w\+\>/
syn match forgatherJinjaVar contained /ns\.\w\+/
syn region forgatherJinjaString contained start=/"/ skip=/\\"/ end=/"/
syn region forgatherJinjaString contained start=/'/ skip=/\\'/ end=/'/
syn match forgatherJinjaFilter contained /|\w\+/
syn match forgatherJinjaFunction contained /\w\+\s*(/

" Brackets and parentheses in Jinja expressions - fix parentheses matching
syn region forgatherJinjaBrackets contained start=/\[/ end=/\]/ contains=forgatherJinjaString,forgatherJinjaVar,forgatherJinjaBrackets,forgatherJinjaParens
syn region forgatherJinjaParens contained start=/(/ end=/)/ contains=forgatherJinjaString,forgatherJinjaVar,forgatherJinjaParens,forgatherJinjaBrackets

" Include parentheses in relevant contexts
syn cluster forgatherJinjaExpressions contains=forgatherJinjaVar,forgatherJinjaString,forgatherJinjaFilter,forgatherJinjaFunction,forgatherJinjaBrackets,forgatherJinjaParens

" Update line statement regions to include parentheses  
syn region forgatherJinjaStmt start=/\S/ end=/$/ contained contains=forgatherJinjaKeyword,@forgatherJinjaExpressions
syn region forgatherJinjaExpr start=/\S/ end=/$/ contained contains=@forgatherJinjaExpressions

" Template split markers
syn match forgatherTemplateSplit /^#\s*-\{3,\}\s*[\w./]\+\s*-\{3,\}$/

" Custom YAML tags - reorganize for better matching
" Simple tags without import specs
syn match forgatherYamlTagSimple /!\(var\|tuple\|list\|dict\|dlist\)\>/

" Tags with import specifications - handle the full module path properly
syn match forgatherYamlTagWithImport /!\(call\|singleton\|factory\|partial\|lambda\):[^@\s[:space:]]\+\(@\w\+\)\?/ contains=forgatherYamlTag,forgatherImportSpec,forgatherNodeName
syn match forgatherYamlTag contained /!\(call\|singleton\|factory\|partial\|lambda\)/
syn match forgatherImportSpec contained /:[^@[:space:]]\+/ 
syn match forgatherNodeName contained /@\w\+/

" YAML anchors and aliases (enhance base YAML highlighting)
syn match forgatherYamlAnchor /&\w\+/
syn match forgatherYamlAlias /\*\w\+/

" Dot-name elision (keys starting with dot)
syn match forgatherDotKey /^\s*\.\w\+\s*:/ contains=forgatherDotName
syn match forgatherDotName contained /\.\w\+/

" Special context variables
syn match forgatherContextVar /!\var\s\+"\w\+"/
syn region forgatherVarBlock start=/!\var\s*$/ end=/^\s*\S/ contains=forgatherYamlKey,forgatherYamlValue

" Highlight groups
hi def link forgatherLineComment Comment
hi def link forgatherInlineComment Comment
hi def link forgatherJinjaComment Comment
hi def link forgatherLineStmt PreProc
hi def link forgatherLineStmtTrimLeft PreProc  
hi def link forgatherLineStmtTrimRight PreProc
hi def link forgatherPrintStmt PreProc
hi def link forgatherPrintStmtTrim PreProc
hi def link forgatherJinjaPrintDelim PreProc
hi def link forgatherJinjaBlockDelim PreProc
hi def link forgatherJinjaKeyword Keyword
hi def link forgatherJinjaVar Identifier
hi def link forgatherJinjaString String
hi def link forgatherJinjaFilter Function
hi def link forgatherJinjaFunction Function
hi def link forgatherJinjaBrackets Delimiter
hi def link forgatherJinjaParens Delimiter
hi def link forgatherTemplateSplit SpecialComment
hi def link forgatherYamlTag Type
hi def link forgatherYamlTagSimple Type
hi def link forgatherImportSpec Special
hi def link forgatherNodeName Label
hi def link forgatherYamlAnchor Label
hi def link forgatherYamlAlias Constant
hi def link forgatherDotName Special
hi def link forgatherContextVar Special

let b:current_syntax = "forgather"