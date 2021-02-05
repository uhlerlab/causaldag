from causaldag import DAG


cancer_network = DAG(arcs={
    ('Pollution', 'Cancer'),
    ('Smoker', 'Cancer'),
    ('Cancer', 'Xmy'),
    ('Cancer', 'Dysponoea')
})

earthquake_network = DAG(arcs={
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
})

sachs_network = DAG(arcs={
    ('PKC', 'PKA'),
    ('PKC', 'Jnk'),
    ('PKC', 'P38'),
    ('PKC', 'Raf'),
    ('PKC', 'Mek'),
    ('PKA', 'Jnk'),
    ('PKA', 'P38'),
    ('PKA', 'Raf'),
    ('PKA', 'Mek'),
    ('PKA', 'Erk'),
    ('PKA', 'Akt'),
    ('Raf', 'Mek'),
    ('Mek', 'Erk'),
    ('Erk', 'Akt'),
    ('Plcg', 'PIP3'),
    ('Plcg', 'PIP2'),
    ('PIP3', 'PIP2'),
})
