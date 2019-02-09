The second portion of the javascript snippet didn't work for me.
Instead:

Use this to generate a comma delimited list of urls in the urls variable 

urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);

Then set it to a local storage variable localStorage.setItem("bananas", urls)

Copy the local storage value into notepad++ and do a replace all , with \n. Make sure the extended options are enabled

The images may still need to be cleaned up after running the validation function. I had to manually delete ~10 before I was able
to successfully fit a model.

I started getting parallel processing errors a few times while generating the data bunch of doing the intial fit call. Interrupting the kernel resolved the issue. Interrupting the kernel also resolved an issue with lr_find freezing up.


TODO: figure out why the Learner object doesn't have an export method. (had to call from fastai.vision import *)
