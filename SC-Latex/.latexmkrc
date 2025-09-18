$out_dir = 'build';

$pdf_mode = 1;  # pdfLaTeX (default)
$pdflatex = 'pdflatex   -interaction=nonstopmode    -file-line-error    -synctex=1';
$lualatex = 'lualatex   -interaction=nonstopmode    -file-line-error    -synctex=1';
$xelatex = 'xelatex     -interaction=nonstopmode    -file-line-error    -synctex=1';

$bibtex_use = 2;
$biber = 'biber --quiet';

add_cus_dep('glo','gls',0,'makeglossaries');
sub makeglossaries {system "makeglossaries \"$_[0]\""};

$halt_on_error = 0;