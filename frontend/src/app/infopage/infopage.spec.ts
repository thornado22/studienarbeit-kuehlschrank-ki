import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Infopage } from './infopage';

describe('Infopage', () => {
  let component: Infopage;
  let fixture: ComponentFixture<Infopage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Infopage],
    }).compileComponents();

    fixture = TestBed.createComponent(Infopage);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
