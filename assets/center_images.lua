Image = function(e)
  if not e.attributes['fig-align'] then
    e.attributes['fig-align']='center'
  end
  return e
end